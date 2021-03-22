#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <termios.h> 

int openUart(int comport);
int uartInit(int nSpeed, int nBits, char nEvent, int nStop, int fd);
void uartSend(char send_buf[], int length, int fd);
void uartRead(char receive_buf[], int length, int fd);
