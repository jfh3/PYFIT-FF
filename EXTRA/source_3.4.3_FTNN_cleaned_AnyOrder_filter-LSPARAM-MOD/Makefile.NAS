CC = icc
# On all Pleiades processor types:
CFLAGS = -w2 -Wcheck -g -O3 -align -fPIC -std=c++11 -ipo -axCORE-AVX2 -xSSE4.2 -fast
# Only on Sandy Bridge or Ivy Bridge processors:
#CFLAGS = -Wall -g -O3 -align -fPIC -xAVX -ip
LDFLAGS = -lstdc++ -lmpi++ -lmpi -mkl
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = fit

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

-include $(OBJECTS:.o=.d)

.cpp.o:
	$(CC) $(CFLAGS) -c $<
	$(CC) -MM $(CFLAGS) $< > $*.d

clean:
	rm -fv $(OBJECTS) $(EXECUTABLE) *.d
