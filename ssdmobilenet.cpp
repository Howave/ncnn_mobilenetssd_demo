#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <pthread.h>

#include "net.h"

#include <sys/time.h>

static struct timeval tv_begin;
static struct timeval tv_end;
static double elasped;

static void bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

static void bench_end(const char* comment)
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
    fprintf(stderr, "%.2fms   %s\n", elasped, comment);
}

struct Object{
    cv::Rect rec;
    int class_id;
    float prob;
};

const char* class_names[] = {"background",
                            "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"};

std::vector<Object> objects;

static int detect_mobilenet(cv::Mat& raw_img, float show_threshold)
{
    ncnn::Net mobilenet;
    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;
#if 0
    mobilenet.load_param("mobilenet_ssd.param");
    mobilenet.load_model("mobilenet_ssd.bin");
#else
    FILE* fp = fopen("./data/model/ulsDetect.bin", "rb");
    mobilenet.load_param_bin(fp);
    mobilenet.load_model(fp);
    fclose(fp);
#endif
    int input_size = 300;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows, input_size, input_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;

    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(2);
#if 0
    ex.input("data", in);
    ex.extract("detection_out",out);
#else
    ex.input(0, in);
    ex.extract(149,out);
#endif

    printf("w:%d, h:%d, c:%d\n", out.w, out.h, out.c);
    objects.clear();
    for (int iw=0;iw<out.h;iw++)
    {
        Object object;
        const float *values = out.row(iw);
        object.class_id = values[0];
        object.prob = values[1];
        object.rec.x = values[2] * img_w;
        object.rec.y = values[3] * img_h;
        object.rec.width = values[4] * img_w - object.rec.x;
        object.rec.height = values[5] * img_h - object.rec.y;
        if (object.class_id == 15)
        {
            objects.push_back(object);
        }
    }
#if 0
    for(int i = 0;i<objects.size();++i)
    {
        Object object = objects.at(i);
        if(object.class_id ==15)//if(object.prob > show_threshold)
        {
            cv::rectangle(raw_img, object.rec, cv::Scalar(255, 0, 255));
            std::ostringstream pro_str;
            pro_str<<object.prob;
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(raw_img, cv::Rect(cv::Point(object.rec.x, object.rec.y- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(214, 112, 218), CV_FILLED);
            cv::putText(raw_img, label, cv::Point(object.rec.x, object.rec.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
    cv::imshow("result",raw_img);
#endif
    return 0;
}

cv::Mat frame;
int quit = 0;

pthread_cond_t cond;
pthread_mutex_t mutex;
pthread_rwlock_t ssdLock;

pthread_cond_t ssdcond;
pthread_mutex_t ssdmutex;

void * imgShowThread(void * arg)
{
    cv::VideoCapture * capture = (cv::VideoCapture *)arg;

    (*capture) >> frame;
    fprintf(stderr, "width[%d], height[%d].\n", frame.cols, frame.rows);

    while (1)
    {
        pthread_mutex_lock(&mutex);
        (*capture) >> frame;
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mutex);

        for(int i = 0;i<objects.size();++i)
        {
            Object object = objects.at(i);

            cv::rectangle(frame, object.rec, cv::Scalar(255, 0, 255));
            std::ostringstream pro_str;
            pro_str<<object.prob;
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame, cv::Rect(cv::Point(object.rec.x, object.rec.y- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(214, 112, 218), CV_FILLED);
            cv::putText(frame, label, cv::Point(object.rec.x, object.rec.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
        cv::imshow("ulsee",frame);

        int c = cv::waitKey(5);
        if( c == 27 || c == 'q' || c == 'Q' )
        {
            quit = 1;
            break;
        }
    }
    std::cout << " imgshow thread quit" << std::endl;
    return NULL;
}

void * ssdThread(void * arg)
{
    while (!quit)
    {
        pthread_mutex_lock(&mutex);
        pthread_cond_wait(&cond,&mutex);

        pthread_mutex_unlock(&mutex);
        pthread_rwlock_wrlock(&ssdLock);

        bench_start();
        detect_mobilenet(frame,0.5);
        bench_end("ssd");
        pthread_rwlock_unlock(&ssdLock);

    }

    pthread_cond_broadcast(&ssdcond);

    return NULL;
}

int main(int argc, char** argv)
{
    if (argc > 1) {
        const char* imagepath = argv[1];

        cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }
        detect_mobilenet(m,0.5);
        cv::waitKey(0);
    } else {
        cv::VideoCapture capture;
        if ( !capture.open(0))
        {
            printf("open video device /dev/video%d fail\n", 1);
            return -1;
        }
        if( capture.isOpened() )
        {
            pthread_t  imgTid, ssdTid;
            void * ssdThreadRe;

            pthread_cond_init(&cond, NULL);
            pthread_mutex_init(&mutex, NULL);

            pthread_cond_init(&ssdcond, NULL);
            pthread_mutex_init(&ssdmutex, NULL);

            pthread_create(&imgTid, NULL, imgShowThread, (void *)(&capture));
            pthread_create(&ssdTid, NULL, ssdThread, NULL);

            pthread_join(ssdTid, &ssdThreadRe);
        }
    }

    return 0;
}
