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
