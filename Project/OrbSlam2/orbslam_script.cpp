#include<KeyFrame.h>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<unistd.h>
#include <stdlib.h>
#include<opencv2/core/core.hpp>
#include <Converter.h>	
#include<System.h>
#include "cv.h" 
#include "highgui.h"


#include <atomic>
#include <csignal>
#include <thread>
#include <pthread.h>
#include <semaphore.h>
#include <opencv2/core/cvstd.hpp>

#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>

// 640*480
#define WIDTH 640
#define HEIGHT 480

// RGB
#define NUMCOLORS 3

#define BUFFER_SIZE 5


void saveMap(ORB_SLAM2::System* SLAM);
void init_vars();
//void pose_data_generator();



typedef struct threads{
	std::unique_ptr<std::thread> socket_from_py;
	std::unique_ptr<std::thread> frame_producer;
	std::unique_ptr<std::thread> pose_producer;
}threads_t;


// start numbering at 1 instead of 0
typedef enum {
	END = 1,
	SAVEMAP

} protocol_messages_t;


namespace serversock {
    
    struct objectData {
        unsigned int value;
    };
    
    void createConnection();
    int readValues(objectData *a);
}



using namespace std;
using namespace serversock;

/*
https://stackoverflow.com/questions/33875583/call-python-from-c-opencv-3-0
*/
struct frame
{
	uint8_t data[HEIGHT] [WIDTH] [NUMCOLORS];
};

// global variables
sem_t full,empty;
pthread_mutex_t mutex1;
pthread_mutex_t mutex2;
int in=0;
int out=0; 
cv::Ptr<cv::Mat> buffer[BUFFER_SIZE];
std::string fstrPointData;
std::atomic<bool> continue_slam(true);
threads_t threads;
ORB_SLAM2::System* pSLAM;

double consume_time = 1;
double produce_time = 1;


void socket_listener()
{
	int counter = 0;
	struct serversock::objectData data;

	objectData *pointer = &data;
	pointer->value = 0;
    serversock::createConnection();
    while (continue_slam) {
        serversock::readValues(pointer);
        if (pointer->value != 0)
        {
        	protocol_messages_t val = static_cast<protocol_messages_t>(pointer->value);
        	switch (val){
        		case END: continue_slam = false; sem_post(&empty); break;
    			case SAVEMAP: saveMap(pSLAM); break;
    			default: std::cout << "Unknown message recived" << std::endl;
        	}
        }
        pointer->value = 0; 
    }
}


/*
case SEND_POSE_DATA: if (counter==0){
    				threads.pose_producer = std::unique_ptr<std::thread>(new std::thread(pose_data_generator)); threads.pose_producer->detach();
    				counter++; break;}
*/
/*
void pose_data_generator()
{
	std::cout << "In pose_data_generator" << std::endl;
	while(continue_slam)
	{
		ORB_SLAM2::Tracking* tracker = pSLAM->GetTracker();
		std::cout << "Referenced tracker" << std::endl;
		//get last frame
    	
		ORB_SLAM2::KeyFrame* pkeyframe = tracker->tracked_frames.front().reference_keyframe;
    	std::cout << "got current frame" << std::endl;
    	cv::Mat cam_center= pkeyframe->GetCameraCenter();
        std::cout << "Last Pose:" << cam_center << std::endl;
	}
}
*/


/* https://github.com/codophobia/producer-consumer-problem-solution-in-c/blob/master/producer-consumer.c */
cv::Mat getframe()
{
	cv::Ptr<cv::Mat> smartPtr;	
	sem_wait(&full);
	pthread_mutex_lock(&mutex1);

	/* CONSUME */
	cv::Mat frame = buffer[out]->clone();
	out = (out + 1) % BUFFER_SIZE;

	pthread_mutex_unlock(&mutex1);
	sem_post(&empty);
	return frame;
} 

void frames_generator()
{
	struct frame fr = {};
	cv::Ptr<cv::Mat> smartPtr;
	int counter = 0;
	while(continue_slam)
	{
		

		sem_wait(&empty);
		pthread_mutex_lock(&mutex1);

		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		/* PRODUCE */
		
		// Reads in the raw data
		do{
			fr = {};
			std::fread(&fr, 1, sizeof(fr.data), stdin);

			// Rebuild raw data to cv::Mat
			smartPtr = new cv::Mat(HEIGHT, WIDTH, CV_8UC3, *fr.data);
		}while(smartPtr->empty() && continue_slam);
		if (continue_slam)
		{
			// before passing into slam system
			cv::cvtColor(*smartPtr, *smartPtr, CV_BGR2RGB);

			buffer[in] = smartPtr;
			in = (in + 1) % BUFFER_SIZE;

			std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
			pthread_mutex_lock(&mutex2);
	        produce_time= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
	        pthread_mutex_unlock(&mutex2);
			pthread_mutex_unlock(&mutex1);
			sem_post(&full);
			counter++;
			//std::cout << "counter is " << counter << std::endl; 
			if (counter > 50)
			{
				sleep(consume_time * 0.95);
			}
		}
		else{
            pthread_mutex_unlock(&mutex1);
			sem_post(&full);
		}
	}
}

int main()
{
   	init_vars();
	double time_stamp;
	int counter = 0;
	std::string home_env_p(std::getenv("HOME"));
	std::cout << "the path is "<< home_env_p << std::endl;
	std::string path_to_vocabulary = home_env_p + "/ORB_SLAM2/Vocabulary/ORBvoc.txt";
	std::string path_to_settings = home_env_p + "/ORB_SLAM2/Examples/Monocular/TUM1.yaml";
	std::string fstrPointData = home_env_p + "/PointData.csv";
	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	ORB_SLAM2::System SLAM(path_to_vocabulary, path_to_settings, ORB_SLAM2::System::MONOCULAR,true);
    ORB_SLAM2::Tracking* ptracker = SLAM.GetTracker();
    pSLAM = &SLAM;
    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;

	/* https://stackoverflow.com/questions/18375998/is-it-possible-to-define-an-stdthread-and-initialize-it-later */
	threads.socket_from_py = std::unique_ptr<std::thread>(new std::thread(socket_listener));
	threads.socket_from_py->detach();
	threads.frame_producer = std::unique_ptr<std::thread>(new std::thread(frames_generator));
	threads.frame_producer->detach();
	

	// Main loop
	while (continue_slam)
	{
		#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
        /*
        cv::imshow( "press ESC on this screen to quit", getframe());
		if ( (cv::waitKey(50) & 255) == 27 ) break; 
		*/
        

		/*
		int empty_val;
		sem_getvalue(&empty, &empty_val);
		int full_val;
		sem_getvalue(&full, &full_val);
		std::cout << "(producer) in= "<< in << ", (producer) full= "<< full_val << std::endl << 
		"(consumer) out= " << out << ", (consumer) empty= "<< empty_val << std::endl << std::endl;
		*/

		/*http://20sep1995.blogspot.com/2019/02/how-to-run-orb-slam-with-your-own-data.html*/
		time_stamp = (0.2) * (counter + 1);

		// Pass the image to the SLAM system
        auto tcw = SLAM.TrackMonocular(getframe(), time_stamp);

		pthread_mutex_lock(&mutex2);
		if (ptracker->mLastProcessedState != ORB_SLAM2::Tracking::NOT_INITIALIZED)
		{

			std::cout << "isSlamInitialized" << true << std::endl; 
	        if (!tcw.empty())
	        {
	        	std::cout << "listPose" << tcw(cv::Range(0,3), cv::Range(3, 4)).t() << std::endl; 
	        }


	        /*
	        try{
	        	//cv::Mat pose = tcw(cv::Range(0,3), cv::Range(3, 4)); // get twc[0:3, 3] - 4th column, up to 3rd row (excluded)
	        	
	        }
	        catch (cv::Exception e){}
			*/
	        std::cout << "isWall" << (int)(ptracker->mLastProcessedState == ORB_SLAM2::Tracking::LOST) << std::endl;
		    std::cout << "isTrackingLost" << (int)(ptracker->mLastProcessedState == ORB_SLAM2::Tracking::LOST) << std::endl;
		    
		}
    	else{
    		std::cout << "isSlamInitialized" << false << std::endl; 
    	}
		pthread_mutex_unlock(&mutex2);

        #ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
        //usleep(0.1);
        /*
        pthread_mutex_lock(&mutex2);
        consume_time= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        std::cout << "(producer) produce_time= " << produce_time << std::endl <<
        "(consumer) consume_time= " << consume_time << std::endl << std::endl;
        pthread_mutex_unlock(&mutex2);
        counter++;
 		*/
        /*
		// Wait to load the next frame
		double T = (0.4) * (counter + 2) - time_stamp;
        if(ttrack<T)
            usleep((T-ttrack)*1e6);
		*/
	}
	/*
	std::cout << "Hi I am Here, joining threads" << std::endl;
	threads.frame_producer->join();
	std::cout << "Hi I am Here, not quite done joining threads" << std::endl;
	threads.socket_from_py->join();
	if (threads.pose_producer != nullptr)
		threads.pose_producer->join();
	*/

	std::cout << "Hi I am Here, done joining threads" << std::endl;
	// Stop all threads
    SLAM.Shutdown();
    std::cout << "Hi I am Here, Slam is shutdown" << std::endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
   	return 0;
}



void saveMap(ORB_SLAM2::System* SLAM){

    std::vector<ORB_SLAM2::MapPoint*> mapPoints = SLAM->GetMap()->GetAllMapPoints();
    std::ofstream pointData;
    pointData.open("/tmp/pointData.csv");
    for(auto p : mapPoints) {
        if (p != NULL)
        {
            auto point = p->GetWorldPos();
            Eigen::Matrix<double, 3, 1> v = ORB_SLAM2::Converter::toVector3d(point);
            pointData << v.x() << "," << v.y() << "," << v.z()<<  std::endl;
        }
    }
    pointData.close();
    pthread_mutex_lock(&mutex2);
    std::cout << "isMapSaved" << true << std::endl;
    pthread_mutex_unlock(&mutex2);
}


void init_vars()
{
	sem_init(&empty,0,BUFFER_SIZE);
	sem_init(&full,0,0);
	pthread_mutex_init(&mutex1,NULL);
	pthread_mutex_init(&mutex2,NULL);
} 




#define PORT "8080"
#define IP "127.0.0.1"



int sockfd, n;
struct sockaddr_in serv_addr;
char socket_buffer[256];

void serversock::createConnection() {
    
    /* Create a socket point */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
    if (sockfd < 0) {
        perror("ERROR opening socket");
        exit(1);
    } else if (sockfd > 0) {
        cout << "SOCKET OPENED" << endl;
    }
    
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(atoi(PORT));
    inet_pton(AF_INET, IP, &(serv_addr.sin_addr.s_addr));
    
    cout << "attempting to connect to server" << endl;
    
    int conn_success = connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    
    if (conn_success < 0) {
        perror("ERROR connecting");
    } else {
        cout << "connection successful" << endl;
    }
    
}

int serversock::readValues(objectData *a) {
    
    fd_set fds;
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    
    
    FD_ZERO(&fds);
    FD_SET(sockfd, &fds);
    select(sockfd+1, &fds, NULL, NULL, &tv);
    
    if (FD_ISSET(sockfd, &fds)) {
        /* The socket_fd has data available to be read */
        n = recv(sockfd, socket_buffer, sizeof(socket_buffer), 0);
        if (n != sizeof(struct objectData)) {
            return 0;
        }
        cout << "Client received: " << n << endl;
        struct objectData data = *((struct objectData *) socket_buffer);
        *a = *((struct objectData *) socket_buffer);
        cout << a->value << endl;
    } 
    /*else {
        cout << "nothing received" << endl;
    }
    */
    
    return 0;
}
