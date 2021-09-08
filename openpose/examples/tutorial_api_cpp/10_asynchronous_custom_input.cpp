// ------------------------- OpenPose C++ API Tutorial - Example 10 - Custom Input -------------------------
// Asynchronous mode: ideal for fast prototyping when performance is not an issue.
// In this function, the user can implement its own way to create frames (e.g., reading his own folder of images)
// and emplaces/pushes the frames to OpenPose.

// Third-party dependencies
#include <opencv2/opencv.hpp>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <darknet.h>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER

#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
DEFINE_string(image_dir,
"examples/media/",
"Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");

using namespace cv;
using namespace std;

void configureWrapper(op::Wrapper &opWrapper) {
    try {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
                0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority) FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution),
                                                       "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution),
                                                       "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                    "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                    " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
                poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
                FLAGS_num_gpu_start, FLAGS_scale_number, (float) FLAGS_scale_gap,
                op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
                (float) FLAGS_alpha_pose, (float) FLAGS_alpha_heatmap, FLAGS_part_to_show,
                op::String(FLAGS_model_folder),
                heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float) FLAGS_render_threshold,
                FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
                op::String(FLAGS_caffemodel_path), (float) FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
                FLAGS_face, faceDetector, faceNetInputSize,
                op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
                (float) FLAGS_face_alpha_pose, (float) FLAGS_face_alpha_heatmap, (float) FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
                FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float) FLAGS_hand_scale_range,
                op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose),
                (float) FLAGS_hand_alpha_pose,
                (float) FLAGS_hand_alpha_heatmap, (float) FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
                FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
                FLAGS_cli_verbose, op::String(FLAGS_write_keypoint),
                op::stringToDataFormat(FLAGS_write_keypoint_format),
                op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
                FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
                op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
                op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format),
                op::String(FLAGS_write_video_3d),
                op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
                op::String(FLAGS_udp_port)};
        opWrapper.configure(wrapperStructOutput);
        // GUI (comment or use default argument to disable any visual output)
        const op::WrapperStructGui wrapperStructGui{
                op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
        opWrapper.configure(wrapperStructGui);
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
//        if (FLAGS_disable_multi_thread)
        opWrapper.disableMultiThreading();
    }
    catch (const std::exception &e) {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

cv::Mat cropping(cv::Mat img) {
    int offset_x = img.size().width/2;
    int offset_y = img.size().height/2-275;
    cv::Rect roi;
    roi.x = offset_x;
    roi.y = offset_y;
    roi.width = 600;
    roi.height = 600;

    /* Crop the original image to the defined ROI */
    cv::Mat crop = img(roi);
    //cv::imshow("crop", crop);

    return crop;
}

void darknet(cv::Mat inputConverted, network *net, image **alphabet,char **labels,  size_t classes) {
    float thresh = 0.7;
    float hier_thresh = 0.25;
    float nms = 0.3;

    cv::Mat darknetConverted;
    cv::cvtColor(inputConverted, darknetConverted, cv::COLOR_BGR2RGB);

    // Convert freenect frame to darknet
    int i, j, k;
    int w = darknetConverted.size().width;
    int h = darknetConverted.size().height;
    int c = darknetConverted.channels();
    image im = make_image(w, h, c);
    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int dst_index = i + w * j + w * h * k;
                int src_index = k + c * i + c * w * j;
                im.data[dst_index] = (float) darknetConverted.data[src_index] / 255.0;
            }
        }
    }
    image sized = letterbox_image(im, net->w, net->h);
    float *frame_data = sized.data;

    // darknet + yolov4 ---------------------
    network_predict(*net, frame_data);
    // ---------------------

    //            // darknet + yolov3 ---------------------
    //            network_predict(net, frame_data);
    //            // ---------------------

    int num_boxes = 0;

    // darknet + yolov4 ---------------------
    detection *detections = get_network_boxes(net, sized.w, sized.h, thresh, hier_thresh, &num_boxes, 1, &num_boxes, 0);
    // ---------------------

    //            // darknet + yolov3 ---------------------
    //            detection *detections = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, &num_boxes, 1, &num_boxes);
    //            // ---------------------

    if (nms > 0) do_nms_obj(detections, num_boxes, classes, nms);
    //
    //            // darknet + yolov4 ---------------------
    draw_detections_old(im, detections, num_boxes, thresh, labels, nullptr , classes);
    show_image(im, "darknet");
    // ---------------------

    // darknet + yolov3 ---------------------
    //            draw_detections(im, detections, num_boxes, thresh, labels, alphabet, classes);
    //            show_image(im, "darknet", 1);
    // ---------------------
    free_detections(detections, num_boxes);
    free_image(im);
    free_image(sized);
}

void openposeTracking(op::Wrapper &opWrapper, cv::Mat frame) {
    // Create new openpose datum
    auto datumsPtr = std::make_shared < std::vector < std::shared_ptr < op::Datum>>>();
    datumsPtr->emplace_back();
    auto &datumPtr = datumsPtr->at(0);
    datumPtr = std::make_shared<op::Datum>();

    // Convert cv mat to openpose
    datumPtr->cvInputData = OP_CV2OPCONSTMAT(frame);

    if (datumPtr != nullptr) {
        auto successfullyEmplaced = opWrapper.waitAndEmplace(datumsPtr);
        if (!successfullyEmplaced)
            op::opLog("Processed datum could not be emplaced.", op::Priority::High);
    }
}

void playVideo(network *net, image **alphabet,char **labels,  size_t classes, op::Wrapper &opWrapper) {
    VideoCapture cap("rgb1628859159.avi");
    //CvCapture *cap = cvCreateFileCapture("/home/kyra/openpose/rgb1628855511.avi"q);

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return;
    }
    while (1) {
        Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        // Display the resulting frame

        cv::Mat crop = cropping(frame);
        imshow("Frame", crop);
        darknet(crop, net, alphabet,labels, classes);

        openposeTracking(opWrapper, crop);
        // Press  ESC on keyboard to exit
        cv::waitKey(10);
//        char c = (char)cv::waitKey(0);
//        std::cout << (int)c << std::endl;
//        if (c == 27)
//            break;


    }
    // When everything done, release the video capture object
    cap.release();
    // Closes all the frames
    destroyAllWindows();
}

void playFreenect(network *net, image **alphabet,char **labels,  size_t classes, op::Wrapper opWrapper){
    // Configure libfreenect
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    std::string serial = freenect2.getDefaultDeviceSerialNumber();
    dev = freenect2.openDevice(serial);

    int types = 0;
    types |= libfreenect2::Frame::Color;
    types |= libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
    libfreenect2::SyncMultiFrameListener listener(types);
    libfreenect2::FrameMap frames;

    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);
    dev->start();

    libfreenect2::Registration *registration = new libfreenect2::Registration(dev->getIrCameraParams(),
                                                                              dev->getColorCameraParams());
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);
    unsigned char *rgbCopy = nullptr;
    unsigned char *depthCopy = nullptr;
    bool userWantsToExit = false;

    long frameCount = 0;
    //TODO ein array oder vector anlegen in dem die dpth data gepackt werden kann

    unsigned long t = (unsigned long) time(NULL);

    VideoWriter video("rgb" + std::to_string(t) + ".avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                      Size(1920, 1080));
    VideoWriter video1("depth" + std::to_string(t) + ".avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                       Size(512, 424));


    // camera frame loop
    while (!userWantsToExit) {
        // Poll frame from freenect
        if (!listener.waitForNewFrame(frames, 10 * 1000)) // 10 sconds
            {
            std::cout << "timeout!" << std::endl;
            break;
            }
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        registration->apply(rgb, depth, &undistorted, &registered);


        // Extact image data fom frame
        auto frameDataSize = rgb->width * rgb->height * rgb->bytes_per_pixel;
        auto frameDataSizeDepth = depth->width * depth->height * depth->bytes_per_pixel;
        if (rgbCopy != nullptr) delete[] rgbCopy;
        if (depthCopy != nullptr) delete[] depthCopy;
        rgbCopy = new unsigned char[frameDataSize];
        depthCopy = new unsigned char[frameDataSizeDepth];
        std::copy(rgb->data, rgb->data + rgb->width * rgb->height * rgb->bytes_per_pixel, rgbCopy);
        std::copy(depth->data, depth->data + depth->width * depth->height * depth->bytes_per_pixel, depthCopy);

        //TODO analog depth->data kopieren und in das array speichern

        // Convert freenect frame to cv mat
        const cv::Mat cvInputData = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgbCopy);
        cv::Mat inputConverted;
        cv::cvtColor(cvInputData, inputConverted, cv::COLOR_BGRA2BGR);

        //const cv::Mat cvInputDepthData = cv::Mat(depth->height, depth->width, CV_8UC4, depthCopy);
        video.write(inputConverted);
        //video1.write(cvInputDepthData);

        darknet(inputConverted, net, alphabet,labels, classes);

        // Create new openpose datum
        auto datumsPtr = std::make_shared < std::vector < std::shared_ptr < op::Datum>>>();
        datumsPtr->emplace_back();
        auto &datumPtr = datumsPtr->at(0);
        datumPtr = std::make_shared<op::Datum>();

        // Convert cv mat to openpose
        datumPtr->cvInputData = OP_CV2OPCONSTMAT(inputConverted);

        if (datumPtr != nullptr) {
            auto successfullyEmplaced = opWrapper.waitAndEmplace(datumsPtr);
            if (!successfullyEmplaced)
                op::opLog("Processed datum could not be emplaced.", op::Priority::High);
        }

        cv::Mat counterImg = cv::Mat::zeros(cv::Size(500, 50), CV_8UC3);
        cv::putText(counterImg, std::to_string(frameCount), cv::Point(10, counterImg.rows / 2),
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0, CV_RGB(118, 185, 0), 2);
        cv::imshow("FrameCounter", counterImg);

        // Cleanup
        listener.release(frames);
        inputConverted.release();
        frameCount++;
    }

    video.release();
    dev->stop();
    dev->close();
}


int tutorialApiCpp() {
    try {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::AsynchronousIn};
        configureWrapper(opWrapper);

        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // Configure darknet and yolo
//        static char *cfg_file = const_cast<char *>("/home/kyra/openpose_dev/yolov4_darknet/cfg/yolov4.cfg");
//        static char *weight_file = const_cast<char *>("/home/kyra/openpose_dev/yolov4_darknet/yolov4.weights");
//        static char *names_file = const_cast<char *>("/home/kyra/openpose_dev/yolov4_darknet/cfg/coco.names");
        static char *cfg_file = const_cast<char *>("/home/kyra/HAR_framework/custom_yolo/custom-yolov4-detector.cfg");
        static char *weight_file = const_cast<char *>("/home/kyra/HAR_framework/custom_yolo/custom-yolov4-detector_final.weights");
        static char *names_file = const_cast<char *>("/home/kyra/HAR_framework/custom_yolo/obj.names");

        size_t classes = 0;
        image **alphabet = load_alphabet();
        char **labels = get_labels(names_file);
        while (labels[classes] != nullptr) {
            classes++;
        }
        std::cout << "Num of Classes " << classes << std::endl;
        network *net = load_network(cfg_file, weight_file, 0);
        set_batch_network(net, 1);

        //playFreenect(net, alphabet, labels, classes, opWrapper);
        playVideo(net, alphabet, labels, classes, opWrapper);
        //video1.release();
        op::opLog("Stopping thread(s)", op::Priority::High);
        opWrapper.stop();


        free(labels);
        //TODO hier die gesammelten daten in einen file schreiben

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception &) {
        return -1;
    }
}

int main(int argc, char *argv[]) {
    // Parsing command line flags
    for(int i = 1; i < argc; i++)
        std::cout << "original arg " << i << " = " << argv[i] << std::endl;

    int new_argc = argc + 2;
    char** new_argv = new char*[new_argc];
    new_argv[argc + 1] = nullptr;
    for (int ii = 0; ii < argc; ++ii) {
        new_argv[ii] = argv[ii];
    }
    new_argv[argc] = new char[strlen("--write_json") + 1]; // extra char for null-terminated string
    strcpy(new_argv[argc], "--write_json");

    unsigned long t = (unsigned long) time(NULL);
    std:string timestring = "testoutput_json_" + std::to_string(t) + "/";
    new_argv[argc+1] = new char[strlen(timestring.c_str()) + 1]; // extra char for null-terminated string
    strcpy(new_argv[argc+1], timestring.c_str());

    for(int i = 1; i < new_argc; i++)
        std::cout << "new arg " << i << " = " << new_argv[i] << std::endl;

    gflags::ParseCommandLineFlags(&new_argc, &new_argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
