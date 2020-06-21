#include <time.h>

class Timer {
public:
	Timer();
	void start();
	void finish();
	void reset();
	double getElapsedTimeMs();

private:
	clock_t begin, end;
};


Timer::Timer() {
	reset();
}

void Timer::reset() {
	begin = 0;
	end = 0;
}

void Timer::start() {
	begin = clock();
}

void Timer::finish() {
	end = clock();
}

double Timer::getElapsedTimeMs() {
	double e = ((double)(end - begin)) / CLOCKS_PER_SEC;
	return e * 1000;
}