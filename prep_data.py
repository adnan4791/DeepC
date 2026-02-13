
import sys

def main():
    width_hr = 352
    height_hr = 288
    
    # Input expected to be CIF
    y_size = width_hr * height_hr
    uv_size = (width_hr // 2) * (height_hr // 2)
    # Output QCIF
    width_lr = 176
    height_lr = 144
    
    input_path = 'akiyo_cif.yuv'
    output_path = 'akiyo_qcif.yuv'
    
    print(f"Reading {input_path}...")
    try:
        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            frame_idx = 0
            while True:
                # Read Y
                y_data = fin.read(y_size)
                if len(y_data) < y_size:
                    break
                
                # Read U
                u_data = fin.read(uv_size)
                # Read V
                v_data = fin.read(uv_size)
                
                # Downsample Y
                # Take every 2nd row and every 2nd pixel
                y_lr = bytearray()
                for r in range(0, height_hr, 2):
                    row_start = r * width_hr
                    # Slice with step 2
                    y_lr.extend(y_data[row_start : row_start + width_hr : 2])
                
                # Downsample U
                u_lr = bytearray()
                for r in range(0, height_hr // 2, 2):
                    row_start = r * (width_hr // 2)
                    u_lr.extend(u_data[row_start : row_start + (width_hr // 2) : 2])

                # Downsample V
                v_lr = bytearray()
                for r in range(0, height_hr // 2, 2):
                    row_start = r * (width_hr // 2)
                    v_lr.extend(v_data[row_start : row_start + (width_hr // 2) : 2])
                
                fout.write(y_lr)
                fout.write(u_lr)
                fout.write(v_lr)

                frame_idx += 1
                if frame_idx % 10 == 0:
                    print(f"Processed frame {frame_idx}")
                    
                if frame_idx >= 50:
                    break
        print(f"Done. Saved to {output_path}")
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")

if __name__ == '__main__':
    main()
