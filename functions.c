#include "math.h"

double cDist(double x1, double x2, double x3, 
            double y1, double y2, double y3) {
    double res = sqrt((x1 - y1) * (x1 - y1) +
                     (x2 - y2) * (x2 - y2) + 
                     (x3 - y3) * (x3 - y3));
    return res;
}

