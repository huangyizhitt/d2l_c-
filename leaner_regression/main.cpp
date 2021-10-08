#include "leaner_regression.h"


int main(int argc, char **argv)
{
    std::vector<float> true_w = {-2, 3.4};
    float true_b = 4.2;
    LR lr(true_w, true_b);
    lr.thensetic_data();
    lr.print_data();
    lr.shuffle_data();
    lr.init_parameter();

    lr.print_model();
    lr.training();
    return 0;
}

