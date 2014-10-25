all: lane_detect.o main.cpp
	g++ -std=c++0x -g lane_detect.o main.cpp -o main `pkg-config --cflags --libs opencv`

lane_detect.o: laneDetection.cpp laneDetection.h
	g++ -std=c++0x -c laneDetection.cpp -o lane_detect.o

lane_detect: laneDetection.cpp laneDetection.h
	g++ -std=c++0x -D LaneTest laneDetection.cpp -o lane_detect `pkg-config --cflags --libs opencv`

clean:
	rm -f main lane_detect *.o 