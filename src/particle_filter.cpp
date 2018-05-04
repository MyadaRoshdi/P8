


#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include "particle_filter.h"

using namespace std;

const int NUMBER_OF_PARTICLES = 50; //300
const double INITIAL_WEIGHT = 1.0;

/***************************************************************
* Set the number of particles. Initialize all particles to first position
* (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
* random gaussian noise is added to each particle
***************************************************************/
void ParticleFilter::init(double x, double y, double theta, double std[]) {

	this->num_particles = NUMBER_OF_PARTICLES;

	
	default_random_engine gen;
	normal_distribution<double> particle_x(x, std[0]);
	normal_distribution<double> particle_y(y, std[1]);
	normal_distribution<double> particle_theta(theta, std[2]);

	for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {

		Particle p = {
			i,
			particle_x(gen),
			particle_y(gen),
			particle_theta(gen),
			INITIAL_WEIGHT
		};

		this->weights.push_back(INITIAL_WEIGHT);
		this->particles.push_back(p);
	}

	this->is_initialized = true;
}


/***************************************************************
*  Add measurements to each particle and add random Gaussian noise.
***************************************************************/
void ParticleFilter::prediction(double delta_t, double std[], double velocity, double yaw_rate) {

	const double THRESH = 0.001;
	const bool MOVING_STRAIGHT = fabs(yaw_rate) < THRESH;
	const double k = MOVING_STRAIGHT ? velocity * delta_t : velocity / yaw_rate;
	const double delta_theta = yaw_rate * delta_t;

	//random_device rdevice;
	//mt19937 gen(rdevice());
	default_random_engine gen;
	normal_distribution<double> nx(0.0, std[0]);
	normal_distribution<double> ny(0.0, std[1]);
	normal_distribution<double> ntheta(0.0, std[2]);

	for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {

		const double theta = this->particles[i].theta;
		const double sin_theta = sin(theta);
		const double cos_theta = cos(theta);
		const double noise_x = nx(gen);
		const double noise_y = ny(gen);
		const double noise_theta = ntheta(gen);

		if (MOVING_STRAIGHT) {

			this->particles[i].x += k * cos_theta + noise_x;
			this->particles[i].y += k * sin_theta + noise_y;
			this->particles[i].theta += noise_theta;

		}
		else {

			const double phi = theta + delta_theta;
			this->particles[i].x += k * (sin(phi) - sin_theta) + noise_x;
			this->particles[i].y += k * (cos_theta - cos(phi)) + noise_y;
			this->particles[i].theta = phi + noise_theta;
		}
	}
}


/**************************************************************
* Find the predicted measurement that is closest to each observed measurement
* and assign the observed measurement to this particular landmark.
***************************************************************/
void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {

	const double BIG_NUMBER = 1.0e99;

	for (int i = 0; i < observations.size(); i++) {

		int current_j;
		double current_smallest_error = BIG_NUMBER;

		for (int j = 0; j < predicted.size(); j++) {

			const double dx = predicted[j].x - observations[i].x;
			const double dy = predicted[j].y - observations[i].y;
			const double error = dx * dx + dy * dy;

			if (error < current_smallest_error) {
				current_j = j;
				current_smallest_error = error;
			}
		}
		observations[i].id = current_j;
	}
}


/***************************************************************
*  Update the weights of each particle using a mult-variate Gaussian distribution.
*  NOTE: The observations are given in the VEHICLE'S coordinate system. Particles are located
*        according to the MAP'S coordinate system. So transformation is done.
* For each particle:
*   1. transform observations from vehicle to map coordinates assuming it's the particle observing
*   2. find landmarks within the particle's range
*   3. find which landmark is likely being observed based on `nearest neighbor` method
*   4. determine the weights based on the difference particle's observation and actual observation
***************************************************************/
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations,
	const Map &map_landmarks) {

	// constants used later for calculating the new weights
	const double stdx = std_landmark[0];
	const double stdy = std_landmark[1];
	const double na = 0.5 / (stdx * stdx);
	const double nb = 0.5 / (stdy * stdy);
	const double d = sqrt(2.0 * M_PI * stdx * stdy);

	for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {

		const double px = this->particles[i].x;
		const double py = this->particles[i].y;
		const double ptheta = this->particles[i].theta;

		vector<LandmarkObs> landmarks_in_range;
		vector<LandmarkObs> map_observations;

		/**************************************************************
		* STEP 1:
		* transform each observations to map coordinates
		* assume observations are made in the particle's perspective
		**************************************************************/
		for (int j = 0; j < observations.size(); j++) {

			const int oid = observations[j].id;
			const double ox = observations[j].x;
			const double oy = observations[j].y;

			const double transformed_x = px + ox * cos(ptheta) - oy * sin(ptheta);
			const double transformed_y = py + oy * cos(ptheta) + ox * sin(ptheta);

			LandmarkObs observation = {
				oid,
				transformed_x,
				transformed_y
			};

			map_observations.push_back(observation);
		}

		/**************************************************************
		* STEP 2:
		* Find map landmarks within the sensor range
		**************************************************************/
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			const int mid = map_landmarks.landmark_list[j].id_i;
			const double mx = map_landmarks.landmark_list[j].x_f;
			const double my = map_landmarks.landmark_list[j].y_f;

			const double dx = mx - px;
			const double dy = my - py;
			const double error = sqrt(dx * dx + dy * dy);

			if (error < sensor_range) {

				LandmarkObs landmark_in_range = {
					mid,
					mx,
					my
				};

				landmarks_in_range.push_back(landmark_in_range);
			}
		}

		/**************************************************************
		* STEP 3:
		* Associate landmark in range (id) to landmark observations
		* this function modifies std::vector<LandmarkObs> observations
		* NOTE: - all landmarks are in map coordinates
		*       - all observations are in map coordinates
		**************************************************************/
		this->dataAssociation(landmarks_in_range, map_observations);

		/**************************************************************
		* STEP 4:
		* Compare each observation (by actual vehicle) to corresponding
		* observation by the particle (landmark_in_range)
		* update the particle weight based on this
		**************************************************************/
		double w = INITIAL_WEIGHT;

		for (int j = 0; j < map_observations.size(); j++) {

			const int oid = map_observations[j].id;
			const double ox = map_observations[j].x;
			const double oy = map_observations[j].y;

			const double predicted_x = landmarks_in_range[oid].x;
			const double predicted_y = landmarks_in_range[oid].y;

			const double dx = ox - predicted_x;
			const double dy = oy - predicted_y;

			const double a = na * dx * dx;
			const double b = nb * dy * dy;
			const double r = exp(-(a + b)) / d;
			w *= r;
		}

		this->particles[i].weight = w;
		this->weights[i] = w;
	}
}


/**************************************************************
* Resample particles with replacement with probability proportional to their weight.
***************************************************************/
void ParticleFilter::resample() {

	vector<Particle> resampled_particles;

	//random_device rdevice;
	//mt19937 gen(rdevice());
	default_random_engine gen;
	discrete_distribution<int> index(this->weights.begin(), this->weights.end());

	for (int c = 0; c < NUMBER_OF_PARTICLES; c++) {

		const int i = index(gen);

		Particle p{
			i,
			this->particles[i].x,
			this->particles[i].y,
			this->particles[i].theta,
			INITIAL_WEIGHT
		};

		resampled_particles.push_back(p);
	}

	this->particles = resampled_particles;
}


void ParticleFilter::write(std::string filename) {

	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);

	for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
		dataFile << this->particles[i].x << " " << this->particles[i].y << " " << this->particles[i].theta << "\n";
	}

	dataFile.close();
}
