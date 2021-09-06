#include "car_control.h"
#include "uart.h"
#include <opencv2/opencv.hpp>



CarControl::CarControl(int port_index) : port_index_{port_index}
{

}

int CarControl::connect()
{
    fd_ = openUart(port_index_);
    if (fd_ == -1)
    {
        return -1;
    }
    if (uartInit(115200, 8, 'n', 1, fd_) != -1)
        return 0;
    else
        return -1;
}

int CarControl::disconnect()
{
    uartSend(CAR_STOP, 10, fd_);
    close(fd_);
}

void CarControl::track(cv::Rect2d &r)
{
    int mid_x, mid_y, mid_size;
    mid_x = int(r.x + r.width / 2);
    mid_y = int(r.y + r.height / 2);
    mid_size = int(r.width);

    int x, y, z;
    char command[11] = CAR_STOP;
    // for safety; if no object
    if (mid_x == 0 && mid_y == 0) 
    {
        uartSend(command, 10, fd_);
        return;
    }
    x = mid_x - 320; y = STOP_Y - mid_y;
    // if already in the center
    if (abs(x) < MID_SENSIT && abs(y) < MID_SENSIT && mid_size > 140) 
    {
        uartSend(command, 10, fd_);
        return;
    }
    if (x < 0) 
    {
        command[9] = command[9] | 0x04;
        command[9] = command[9] & 0x04;
    }
    if (y < 0) 
    {
        command[9] = command[9] | 0x02;
    }
    x = abs(MAX_SPEED * x / 320);
    z = TURN_SENSIT * x;
    y = abs(MAX_SPEED * y / 240);
    //command[4] = x + 5;
    command[6] = y + 5;
    command[8] = z;
    //printf("(%d, %d, %d)\n", x, y, z);
    uartSend(command, 10, fd_);
}