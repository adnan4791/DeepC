import torch
import torch.nn as nn
import numpy as np

class FSRCNN(nn.Module):
    def __init__(self, scale_factor):
        super(FSRCNN, self).__init__()
        # d=56, s=12, m=4
        self.layer1 = nn.Conv2d(1, 56, kernel_size=5, padding=2)
        self.layer2 = nn.Conv2d(56, 12, kernel_size=1)
        self.map3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.map4 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.map5 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.map6 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.layer7 = nn.Conv2d(12, 56, kernel_size=1)
        
        # Layer 8: Deconv (Stride=scale)
        self.layer8 = nn.ConvTranspose2d(56, 1, kernel_size=9, 
                                        stride=scale_factor, padding=4, 
                                        output_padding=scale_factor-1)
        
        # PReLU dengan koefisien statis
        self.prelu1 = nn.PReLU(56)
        self.prelu2 = nn.PReLU(12)
        self.prelu_m = nn.PReLU(12)
        self.prelu7 = nn.PReLU(56)

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = torch.prelu(x, torch.tensor([-0.8986]).to(x.device)) # Coeff C
        # Layer 2
        x = self.layer2(x)
        x = torch.prelu(x, torch.tensor([0.3236]).to(x.device))
        # Mapping 3-6
        x = torch.prelu(self.map3(x), torch.tensor([0.2288]).to(x.device))
        x = torch.prelu(self.map4(x), torch.tensor([0.2476]).to(x.device))
        x = torch.prelu(self.map5(x), torch.tensor([0.3495]).to(x.device))
        x = torch.prelu(self.map6(x), torch.tensor([0.7806]).to(x.device))
        # Layer 7
        x = self.layer7(x)
        x = torch.prelu(x, torch.tensor([0.0087]).to(x.device))
        # Layer 8 (Tanpa PReLU)
        return self.layer8(x)

def load_txt_precise(path, count):
    # Menggunakan float64 dulu agar presisi
    data = np.fromfile(path, sep=' ', count=count) 
    if len(data) == 0: # Jika sep=' ' gagal, coba deteksi line per line
        data = np.loadtxt(path).flatten()[:count]
    return torch.from_numpy(data).float()

def load_all(model, folder):
    with torch.no_grad():
        # Urutan Weight biasanya Out x In x H x W
        model.layer1.weight.copy_(load_txt_precise(f"{folder}/weights_layer1.txt", 1400).view(56, 1, 5, 5))
        model.layer1.bias.copy_(load_txt_precise(f"{folder}/biasess_layer1.txt", 56))
        
        # Layer 2:(i*num_channels + j) -> Out x In
        model.layer2.weight.copy_(load_txt_precise(f"{folder}/weights_layer2.txt", 672).view(12, 56, 1, 1))
        model.layer2.bias.copy_(load_txt_precise(f"{folder}/biasess_layer2.txt", 12))
        
        # Mapping 3-6
        model.map3.weight.copy_(load_txt_precise(f"{folder}/weights_layer3.txt", 1296).view(12, 12, 3, 3))
        model.map3.bias.copy_(load_txt_precise(f"{folder}/biasess_layer3.txt", 12))
        
        model.map4.weight.copy_(load_txt_precise(f"{folder}/weights_layer4.txt", 1296).view(12, 12, 3, 3))
        model.map4.bias.copy_(load_txt_precise(f"{folder}/biasess_layer4.txt", 12))
        
        model.map5.weight.copy_(load_txt_precise(f"{folder}/weights_layer5.txt", 1296).view(12, 12, 3, 3))
        model.map5.bias.copy_(load_txt_precise(f"{folder}/biasess_layer5.txt", 12))
        
        model.map6.weight.copy_(load_txt_precise(f"{folder}/weights_layer6.txt", 1296).view(12, 12, 3, 3))
        model.map6.bias.copy_(load_txt_precise(f"{folder}/biasess_layer6.txt", 12))
        
        # Layer 7
        model.layer7.weight.copy_(load_txt_precise(f"{folder}/weights_layer7.txt", 672).view(56, 12, 1, 1))
        model.layer7.bias.copy_(load_txt_precise(f"{folder}/biasess_layer7.txt", 56))
        
        # Layer 8: Deconv
        model.layer8.weight.copy_(load_txt_precise(f"{folder}/weights_layer8.txt", 4536).view(56, 1, 9, 9))
        # Spesifik Layer 8 Bias adalah single double
        b8 = load_txt_precise(f"{folder}/biasess_layer8.txt", 1)
        model.layer8.bias.fill_(b8.item())

# ==========================================
# PROCESSING YUV 4:2:0
# ==========================================
def process_yuv(input_file, output_file, weights_path):
    w, h = 176, 144
    model = FSRCNN(scale_factor=2)
    load_all(model, weights_path)
    model.eval()
    
    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        while True:
            data = f_in.read(w * h + (w // 2 * h // 2) * 2)
            if not data: break
            
            # Ambil Y
            y = np.frombuffer(data, dtype=np.uint8, count=w*h).reshape(h, w).copy()
            # Model yang dilatih adalah 0-1
            img_y = torch.from_numpy(y).float().view(1, 1, h, w) / 255.0
            
            with torch.no_grad():
                out_y = model(img_y)
                out_y = (out_y.clamp(0, 1).numpy()[0, 0] * 255.0).astype(np.uint8)
            
            # U & V Repetition
            u = np.frombuffer(data, dtype=np.uint8, count=(w//2)*(h//2), offset=w*h).reshape(h//2, w//2)
            v = np.frombuffer(data, dtype=np.uint8, count=(w//2)*(h//2), offset=w*h + (w//2)*(h//2)).reshape(h//2, w//2)
            
            u_up = u.repeat(2, axis=0).repeat(2, axis=1)
            v_up = v.repeat(2, axis=0).repeat(2, axis=1)
            
            f_out.write(out_y.tobytes())
            f_out.write(u_up.tobytes())
            f_out.write(v_up.tobytes())
    print("Selesai!")

process_yuv("suzie_qcif.yuv", "output_final_352x288.yuv", "./my_weights")