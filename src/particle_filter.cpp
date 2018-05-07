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
		if (yaw_rate !< 0) {
			new_x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			new_y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			new_theta = particles[i].theta + yaw_rate * delta_t;
		}
		else {

			new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);		
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
	//   observed measurement to this particular landmark (Note: all are given in map (global) co-ordinates.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	

	double max_dist_value = 99999999999999; // initialize with big number (infiniti)

	for (int i = 0; i < observations.size(); i++)
	{

		int smallest_j = -1;
		int map_id = -1;
		double smallest_dist = max_dist_value;

		for (int j = 0; j < predicted.size(); j++) {

			// calculate Euclidean distance between predicted landmarks(predicted) and true land marks
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			// find nearest neigbor and associate to the predicted landmark 
			if (distance < smallest_dist) {
				smallest_dist = distance;
				map_id = predicted[j].id;
			}
		}
		observations[i].id = map_id;
	}


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Use the sum of weights to Normalize all weights values as suggested
	double weight_sum = 0.0;

	// loop over all the particles
	for (int i = 0; i < num_particles; i++) {

		

		// get the particle x, y coordinates (In map co-ordinates)
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// create a vector to hold the map landmark locations predicted to be within sensor range of the particle
		vector<LandmarkObs> predictions;

		// loop over each map landmark
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			// get id and x,y coordinates
			float lm_x = map_landmarks.landmark_list[j].x_f;
			float lm_y = map_landmarks.landmark_list[j].y_f;
			int lm_id = map_landmarks.landmark_list[j].id_i;

			// Interested only in landmarks within sensor range of the particle 
			if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {

				// add this in range prediction to the predictions vector
				predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
			}
		}

		// Transform the list of observations from vehicle coordinates to map coordinates
		vector<LandmarkObs> transformed_os;
		for (unsigned int j = 0; j < observations.size(); j++) {
			double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
			double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
			transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
		}

		// use dataAssociation() for the predictions and transformed observations for each particle
		dataAssociation(predictions, transformed_os);

		// reinit weight
		particles[i].weight = 1.0;

		for (unsigned int j = 0; j < transformed_os.size(); j++) {

			// rename observation and associated prediction coordinates
			double o_x, o_y, pr_x, pr_y;
			o_x = transformed_os[j].x;
			o_y = transformed_os[j].y;

			int associated_prediction = transformed_os[j].id;

			// get the x,y coordinates of the prediction associated with the current observation
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == associated_prediction) {
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;
				}
			}

			// calculate weight for this observation with multivariate Gaussian
			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double obs_w = (1 / (2 * M_PI*s_x*s_y)) * exp(-(pow(pr_x - o_x, 2) / (2 * pow(s_x, 2)) + (pow(pr_y - o_y, 2) / (2 * pow(s_y, 2)))));

			// Total observations weight is product of all observation weights
			particles[i].weight *= obs_w;
		}

		weight_sum += particles[i].weight;
	}

	// Normalize all weight values
	for (int k = 0; k < particles.size(); k++)
	{
		particles[k].weight = particles[k].weight / weight_sum;
	}
}

/*
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

*/

void ParticleFilter::resample() 
{
	// Resample particles with replacement with probability proportional to their weight (Using Resample wheel Lesson13-20). 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


	// Create a random generator
	default_random_engine gen;

	// create vector resampled_particles to carry the new list of particles
	vector<Particle> resampled_particles;

	// create a vector W to carry all the current particles weights
	vector<double> W;
	for (int i = 0; i < num_particles; i++)
	{
		W.push_back(particles[i].weight);
	}

	// initialize the index for resampling wheel using uniform distribution (index = U[0.....N-1))
	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	int index = uniintdist(gen);

	// find max weight
	double max_weight = *max_element(W.begin(), W.end());

	// uniform random distribution around the max_weight
	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	double beta = 0.0;

	// Use resampling wheel
	for (int i = 0; i < num_particles; i++)
	{
		beta = beta + unirealdist(gen) * 2.0;
		while (W[index] < beta)
		{
			beta -= W[index];
			index = (index + 1) % num_particles;
		}
		resampled_particles.push_back(particles[index]);
	}

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
