#ifndef CAR_CONTROL_H_
#define CAR_CONTROL_H_

#include "uart.h"
#include <opencv2/opencv.hpp>

#define MID_SENSIT 40
#define TURN_SENSIT 2
#define MAX_SPEED 20
#define STOP_Y 420
#define INTER 10
#define CAR_STOP "\xff\xfe\x01\x00\x00\x00\x00\x00\x00\x00"

class CarControl
{
public:
    CarControl(int port_index);

    int connect();
    int disconnect();
    void track(cv::Rect2d &r);

private:
    int port_index_;
    int fd_;
};

#endif
