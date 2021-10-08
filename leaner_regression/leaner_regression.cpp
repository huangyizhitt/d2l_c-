#include "leaner_regression.h"
#include <chrono>
#include <numeric>
#include <cstdio>
#include <algorithm>

LR::LR(const std::vector<float>& w, const float& b, const float& larate, const int& samples, 
        const int& batch_size, const int& epochs) : true_w(w), true_b(b), lrate(lrate), 
        num_samples(samples), batch_size(batch_size), epochs(epochs) 
{
    data.resize(num_samples);

    for(int i = 0; i < num_samples; i++) {
        data[i].inputs.resize(true_w.size());
    }
    weights.resize(true_w.size());
    tmp_weights.resize(true_w.size());

    seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);
}

void LR::thensetic_data()
{
    std::normal_distribution<float> dis_inputs(0.0, 1.0);
    std::normal_distribution<float> dis_noise(0.0, 0.01);

    for(int i = 0; i < num_samples; i++) {
        for(size_t j = 0; j < weights.size(); j++) {
            data[i].inputs[j] = dis_inputs(generator);
        }
        
        data[i].output = std::inner_product(true_w.begin(), true_w.end(), 
                data[i].inputs.begin(), true_b+dis_noise(generator));
    }
}

void LR::shuffle_data()
{
    shuffle (data.begin(), data.end(), std::default_random_engine(seed));
}

float LR::model(const std::vector<float>& x, const std::vector<float>& w, const float& baise)
{
    return std::inner_product(x.begin(), x.end(), w.begin(), baise);
}

float LR::loss()
{
    float ret = 0;
    int n = num_samples;
    for(int i = 0; i < n; i++) {
        float tmp = data[i].output - model(data[i].inputs, weights, b);
        ret += tmp * tmp;
    }
    ret /= n;
    ret = std::sqrt(ret);
}

float LR::gradient_w(const size_t& begin, const size_t& end, int pos)
{
    float ret = 0;
    size_t n = end-begin;
    for(size_t i = begin; i < end; i++) {
        ret += (model(data[i].inputs, weights, b) - data[i].output) * data[i].inputs[pos];
    }
    ret /= n;
    return ret;
}

float LR::gradient_b(const size_t& begin, const size_t& end)
{
    float ret = 0;
    size_t n = end-begin;
    for(size_t i = begin; i < end; i++) {
        ret += (model(data[i].inputs, weights, b) - data[i].output);
    }
    ret /= n;
    return ret;
}

void LR::update_parameters(const size_t& begin, const size_t& end)
{
    for(int i = 0; i < weights.size(); i++) {
        tmp_weights[i] = weights[i] - gradient_w(begin, end, i);
    }

    for(int i = 0; i < weights.size(); i++) {
        weights[i] = tmp_weights[i];
    }

    b = b - gradient_b(begin, end);
}

void LR::training()
{
    for(int i = 0; i < epochs; i++) {
        int k = num_samples / batch_size;
        std::uniform_int_distribution<int> dis(0, k);
        int begin = dis(generator) * batch_size;
        int end = std::min(begin+batch_size, num_samples); 
        update_parameters(begin, end);
/*        for(int begin = 0; begin < num_samples; begin += batch_size) {
            int end = std::min(begin+batch_size, num_samples);
            update_parameters(begin, end);
        }*/

        float l = loss();
        printf("Epoch: %d, loss: %.3f\n", i, l);
        print_model();
        if(l < 1e-7) break;
    }
}

void LR::init_parameter()
{
    std::normal_distribution<float> dis(0, 0.01);

    for(int i = 0; i < weights.size(); i++) {
        weights[i] = dis(generator);
    }
    b = dis(generator);
}

void LR::print_data()
{
    int nums = std::min(5, num_samples);
    for(int i = 0; i < nums; i++) {
        printf("inputs: ");
        for(size_t j = 0; j < weights.size(); j++) {
            printf("%.2f ", data[i].inputs[j]);
        }
        printf("output: %.2f\n", data[i].output);
    }
}

void LR::print_model()
{
    printf("y=");
    for(int i = 0; i < weights.size(); i++) {
        printf("%.2fx%d+", weights[i], i);
    }
    printf("%.2f\n", b);
}