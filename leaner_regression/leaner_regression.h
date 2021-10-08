#ifndef __LEANER_REGRESSION_H__
#define __LEANER_REGRESSION_H__

#include <vector>
#include <cstdlib>
#include <random>

struct Data {
    float output;
    std::vector<float> inputs;
};

class LR {
public:
    LR(const std::vector<float>& w, const float& b, const float& lrate = 0.01,
    const int& samples = 10000, const int& batch_size = 1000, const int& epochs=20);

    void thensetic_data();
    void shuffle_data();
    void print_data();
    void print_model();
    void update_parameters(const size_t& begin, const size_t& end);
    void training();
    void init_parameter();


    float model(const std::vector<float>& x, const std::vector<float>& w, const float& baise);
    float gradient_w(const size_t& begin, const size_t& end, int pos);
    float gradient_b(const size_t& begin, const size_t& end);
    float loss();

private:
    float true_b;
    float b;
    float lrate;
    int num_samples;    
    int batch_size;
    unsigned seed;
    int epochs;

    std::default_random_engine generator;
    std::vector<float> true_w; 
    std::vector<float> weights;
    std::vector<float> tmp_weights;
    std::vector<Data> data;
};

#endif