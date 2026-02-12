CC = gcc
CFLAGS = -fopenmp -O3 -Wall
TARGET = active_deepc
SRC = source.c
LIBS = -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)

convert:
	docker-compose run --rm ffmpeg -f rawvideo -vcodec rawvideo -s 352x288 -r 30 -pix_fmt yuv420p -i output.yuv -c:v libx264 -preset slow -crf 18 -y output.mp4
	@echo "Conversion complete: output.mp4"

play:
	docker run --rm -it -v "$$PWD:/data" --entrypoint ffplay jrottenberg/ffmpeg:6.1-alpine -f rawvideo -pixel_format yuv420p -video_size 352x288 -framerate 30 /data/output.yuv
