/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// This function uses Sense noisy position data from the simulator to initialize the particles
	// Set the number of particles.
	num_particles = 100;


	
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Create a random generator
	default_random_engine gen;

	// define normal distributions for sensor noise around the sensed values of x,y & theta
	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	//Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = N_x(gen);
		p.y = N_y(gen);
		p.theta = N_theta(gen);
		p.weight = 1.0;

		
		particles.push_back(p);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// This function Predicts the vehicle's next state from previous (noiseless control) data (The passed velocity and yaw_rate), but alo adds noise .
	// Prediction of next state (using motion equations[bycicle motion model]) and then apply gaussian noise to it: https://discussions.udacity.com/t/do-we-still-need-to-add-noise-to-velocity-and-yaw-rate-when-using-the-simulator/294827/2
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// New variables to carry the predicted particle state values
	double new_x; 
	double new_y;
	double new_theta;

	// Create a random generator
	default_random_engine gen;

	

	for (int i = 0; i < num_particles; i++) {

		// calculate new state
		if (fabs(yaw_rate) < 0) {
			new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
		}
		else {
			new_x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			new_y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			new_theta = particles[i].theta + yaw_rate * delta_t;
		}

		// Add noise: define normal distributions for sensor noise around the new predicted values of x,y & theta
		normal_distribution<double> N_x(new_x, std_pos[0]);
		normal_distribution<double> N_y(new_y, std_pos[1]);
		normal_distribution<double> N_theta(new_theta, std_pos[2]);

		// Update the particles with the predicted values
		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//  Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	

	double max_dist_value = 1.0e99; // initialize with big number (infiniti)

	for (int i = 0; i < observations.size(); i++)
	{

		int smallest_j = -1;
		int map_id = -1;
		double smallest_dist = max_dist_value;

		for (int j = 0; j < predicted.size(); j++) {

			// calculate Euclidean distance between predicted landmarks(predicted) and true land marks
			double distance = dist(observations[i].x, observations[i].y, predicted[i].x, predicted[i].y);
			// find nearest neigbor and associate to the predicted landmark 
			if (distance < smallest_dist) {
				smallest_j = j;
				map_id = predicted[i].id;
			}
		}
		observations[i].id = map_id;
	}


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

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

void ParticleFilter::resample() {
	//  Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Create a generator to be used for generating random particles
	default_random_engine gen;

	vector<Particle> resampled_particles;
	//Generate discrete distribution 
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	for (int i = 0; i < particles.size(); i++)
	{
		resampled_particles.push_back(particles[distribution(gen)]);
	}

	//num_particles = resampled_particles.size();
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
