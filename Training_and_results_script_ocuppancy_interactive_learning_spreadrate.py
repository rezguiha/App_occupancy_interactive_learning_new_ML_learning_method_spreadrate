__supervisor__ = 'eng.amayri@gmail.com'
__supervisor__ = 'stephane.ploix@g-scop.grenoble-inp.fr'
__author__ = 'hamdi.rezgui1993@gmail.com'
# This the code of an interactive learning number of occupants present inside an office based on data from various sensors installed.
# It brings a new approach to classical machine learning like Decision tree, Randomforest.... by implementing the spreadrate measure developed in
# G-scope lab which i used to overcome the problem of class size disproportionality in the data problem that can occur in classification problems.
#  The spreadrate implementation overcame every solution that exists for this problem present at the time.


import random
import h358data
import math
from sklearn import tree
import random
import unittest
import pickle
import scipy
import matplotlib.pyplot as plt
from sklearn import linear_model
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import plotly.plotly as py
import plotly.tools as tls


def generate_occupancy_estimator_from_CO2(pset: 'metaoptim.ParameterSet'):

    def occupancy_estimator_from_CO2(times: list, co2in: list, co2cor: list, door: list, window: list):
        concentrationCO2out = pset.val('concentrationCO2out')
        CO2breathProduction = pset.val('CO2breathProduction')
        Volume = pset.val('Volume')
        Q0out = pset.val('Q0out')
        Q0corridor = pset.val('Q0corridor')
        QW = pset.val('QW')
        QD = pset.val('QD')
        estimated_occupancies = list()
        for k in range(len(times) - 1):
            alpha_k = math.exp((Q0out + Q0corridor + QD * door[k] + QW * window[k]) * ((times[k + 1] - times[k]) // 1000) / Volume)
            beta_k = (1 - alpha_k) / (Q0out + Q0corridor + QD * door[k] + QW * window[k])
            estimated_occupancy = max(0, (co2in[k + 1] - alpha_k * co2in[k]) / (CO2breathProduction * beta_k) - ((Q0out + QW * window[k]) * concentrationCO2out + (Q0corridor + QD * door[k]) * co2cor[k]) / CO2breathProduction)
            estimated_occupancies.append(estimated_occupancy)
        estimated_occupancies.append(0)
        return estimated_occupancies
    return occupancy_estimator_from_CO2

class Discretization:

    def __init__(self, *intervals: tuple):
        self.intervals = intervals

    def approximated_value(self, continuous_value: float):
        for interval in self.intervals:
            if interval[0] <= continuous_value <= interval[1]:
                return (interval[0] + interval[1]) / 2
        if continuous_value < self.lower():
            return (interval[0][0] + interval[0][1]) / 2
        else:
            return (interval[-1][0] + interval[-1][1]) / 2

    def level_center(self, k: int):
        return (self.intervals[k][0] + self.intervals[k][1]) / 2

    def level(self, continuous_value: float):
        for k in range(len(self.intervals)):
            if self.intervals[k][0] <= continuous_value <= self.intervals[k][1]:
                return k
        if continuous_value < self.lower():
            return 0
        else:
            return len(self.intervals) - 1

    def __len__(self):
        return len(self.intervals)

    def lower(self):
        return self.intervals[0][0]

    def upper(self):
        return self.intervals[-1][1]


class Recording:

    def __init__(self):
        self.data = list()

    def add_serie(self, name: str, values: list):
        while len(self.data) < len(values):
            self.data.append(dict())
        for k in range(len(values)):
            self.data[k][name] = values[k]

    def append(self, record: list):
        self.data.append(dict(record))

    def serie(self, name: str):
        values = list()
        for k in range(len(self.data)):
            values.append(self.data[k][name])
        return values

    def __iter__(self):
        for record in self.data:
            yield record

    def __getitem__(self, k: int):
        return self.data[k]

    def __len__(self):
        return len(self.data)

    def max_value(self, name: str):
        maximum = self.data[0][name]
        for k in range(len(self.data)):
            maximum = max(maximum, self.data[k][name])
        return maximum

    def min_value(self, name: str):
        minimum = self.data[0][name]
        for k in range(len(self.data)):
            minimum = min(minimum, self.data[k][name])
        return minimum



def decision_tree(training_records: Recording, testing_records: Recording):
    training_inputs, training_output = [], []
    occupancy_discretization = Discretization((0, 0), (0, 1.5), (1.5, 4))
    for record in training_records:
        training_inputs.append([record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']])
        training_output.append(occupancy_discretization.level(record['actual_occupancy']))

    classifier = tree.DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=None)
    classifier.fit(training_inputs, training_output)
    features_importance = classifier.feature_importances_
    testing_errors=[]
    testing_inputs = []
    for record in testing_records:
        test_features = [record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']]
        testing_inputs.append(test_features)
    testing_predicted_output = classifier.predict(testing_inputs)

    k, error, precision_counter,error_class_0,error_class_1,error_class_2,size_class_0,size_class_1,size_class_2= 0, 0, 0, 0, 0, 0, 0, 0, 0
    for record in testing_records:
        error += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
        testing_errors.append(error)
        if occupancy_discretization.level(record['actual_occupancy'])==occupancy_discretization.level(testing_predicted_output[k]):
            precision_counter+=1

        if record['actual_occupancy']==0:
            error_class_0 += abs(occupancy_discretization.level(testing_predicted_output[k]) -occupancy_discretization.level(record['actual_occupancy']))
            size_class_0 +=1
        elif record['actual_occupancy'] == 1:
            error_class_1 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_1 += 1
        else:
            error_class_2 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_2 += 1
        k += 1

    return error / len(testing_records),testing_errors,precision_counter/len(testing_records),error_class_0/size_class_0,error_class_1/size_class_1,error_class_2/size_class_2

def linear_regression(training_records: Recording, testing_records: Recording):
    training_inputs, training_output, training_errors = [], [], []

    for record in training_records:
        #print("values I need ",record['actual_occupancy'],record['acoustic_pressure_dB'],record['estimated_occupancy_from_CO2'])
        training_inputs.append([record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']])
        training_output.append(record['actual_occupancy'])

    classifier = linear_model.LinearRegression()
    classifier.fit(training_inputs, training_output)
    testing_inputs = []
    for record in testing_records:
        test_features = [record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']]
        testing_inputs.append(test_features)
    testing_predicted_output = classifier.predict(testing_inputs)
    k, error, precision_counter = 0, 0, 0
    for record in testing_records:
        error += abs(testing_predicted_output[k] - record['actual_occupancy'])
        if record['actual_occupancy']==testing_predicted_output[k]:
            precision_counter+=1
        k += 1


    return error / len(testing_records),testing_predicted_output,precision_counter/len(testing_records)
def support_vector_machine(training_records: Recording, testing_records: Recording):
    training_inputs, training_output, training_errors = [], [], []
    occupancy_discretization = Discretization((0, 0), (0, 1.5), (1.5, 4))
    for record in training_records:

        #print("values I need ",record['actual_occupancy'],record['acoustic_pressure_dB'],record['estimated_occupancy_from_CO2'])
        training_inputs.append([record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']])
        training_output.append(occupancy_discretization.level(record['actual_occupancy']))
    # training_inputs, training_output = make_classification(n_classes=3, n_informative=6, shuffle=True)
    # for a in training_inputs:
    #     print(a)
    #
    # print(len(training_output))
    # print(len(training_inputs))
    # print(len(training_inputs[0]))

    classifier = SVC(decision_function_shape='ovo', gamma='scale', random_state=0)
    # training_inputs = np.array(training_inputs)
    # training_output = np.array(training_output)
    classifier.fit(training_inputs, training_output)
    classifier.score(training_inputs, training_output)
    testing_inputs = []
    for record in testing_records:
        test_features = [record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']]
        testing_inputs.append(test_features)
    testing_predicted_output = classifier.predict(testing_inputs)
    k, error, precision_counter,error_class_0,error_class_1,error_class_2,size_class_0,size_class_1,size_class_2= 0, 0, 0, 0, 0, 0, 0, 0, 0
    for record in testing_records:
        error += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
        if occupancy_discretization.level(record['actual_occupancy'])==occupancy_discretization.level(testing_predicted_output[k]):
            precision_counter+=1

        if record['actual_occupancy']==0:
            error_class_0 += abs(occupancy_discretization.level(testing_predicted_output[k]) -occupancy_discretization.level(record['actual_occupancy']))
            size_class_0 +=1
        elif record['actual_occupancy'] == 1:
            error_class_1 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_1 += 1
        else:
            error_class_2 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_2 += 1
        k += 1

    return error / len(testing_records),testing_predicted_output,precision_counter/len(testing_records),error_class_0/size_class_0,error_class_1/size_class_1,error_class_2/size_class_2
def naive_bayes(training_records: Recording, testing_records: Recording):
    training_inputs, training_output, training_errors = [], [], []
    occupancy_discretization = Discretization((0, 0), (0, 1.5), (1.5, 4))
    for record in training_records:
        #print("values I need ",record['actual_occupancy'],record['acoustic_pressure_dB'],record['estimated_occupancy_from_CO2'])
        training_inputs.append([record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']])
        training_output.append(occupancy_discretization.level(record['actual_occupancy']))

    classifier = MultinomialNB()
    classifier.fit(training_inputs, training_output)
    testing_inputs = []
    for record in testing_records:
        test_features = [record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']]
        testing_inputs.append(test_features)
    testing_predicted_output = classifier.predict(testing_inputs)
    k, error, precision_counter,error_class_0,error_class_1,error_class_2,size_class_0,size_class_1,size_class_2= 0, 0, 0, 0, 0, 0, 0, 0, 0
    for record in testing_records:
        error += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
        if occupancy_discretization.level(record['actual_occupancy'])==occupancy_discretization.level(testing_predicted_output[k]):
            precision_counter+=1

        if record['actual_occupancy']==0:
            error_class_0 += abs(occupancy_discretization.level(testing_predicted_output[k]) -occupancy_discretization.level(record['actual_occupancy']))
            size_class_0 +=1
        elif record['actual_occupancy'] == 1:
            error_class_1 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_1 += 1
        else:
            error_class_2 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_2 += 1
        k += 1

    return error / len(testing_records),testing_predicted_output,precision_counter/len(testing_records),error_class_0/size_class_0,error_class_1/size_class_1,error_class_2/size_class_2
def random_forest(training_records: Recording, testing_records: Recording):
    training_inputs, training_output, training_errors = [], [], []
    occupancy_discretization = Discretization((0, 0), (0, 1.5), (1.5, 4))
    for record in training_records:
        #print("values I need ",record['actual_occupancy'],record['acoustic_pressure_dB'],record['estimated_occupancy_from_CO2'])
        training_inputs.append([record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']])
        training_output.append(occupancy_discretization.level(record['actual_occupancy']))

    classifier = RandomForestClassifier(n_estimators=100, max_depth=2)
    classifier.fit(training_inputs, training_output)
    testing_inputs = []
    for record in testing_records:
        test_features = [record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']]
        testing_inputs.append(test_features)
    testing_predicted_output = classifier.predict(testing_inputs)
    k, error, precision_counter,error_class_0,error_class_1,error_class_2,size_class_0,size_class_1,size_class_2= 0, 0, 0, 0, 0, 0, 0, 0, 0
    for record in testing_records:
        error += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
        if occupancy_discretization.level(record['actual_occupancy'])==occupancy_discretization.level(testing_predicted_output[k]):
            precision_counter+=1

        if record['actual_occupancy']==0:
            error_class_0 += abs(occupancy_discretization.level(testing_predicted_output[k]) -occupancy_discretization.level(record['actual_occupancy']))
            size_class_0 +=1
        elif record['actual_occupancy'] == 1:
            error_class_1 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_1 += 1
        else:
            error_class_2 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_2 += 1
        k += 1

    return error / len(testing_records),testing_predicted_output,precision_counter/len(testing_records),error_class_0/size_class_0,error_class_1/size_class_1,error_class_2/size_class_2
def Neural_Network(training_records: Recording, testing_records: Recording):
    training_inputs, training_output, training_errors = [], [], []
    occupancy_discretization = Discretization((0, 0), (0, 1.5), (1.5, 4))
    for record in training_records:
        #print("values I need ",record['actual_occupancy'],record['acoustic_pressure_dB'],record['estimated_occupancy_from_CO2'])
        training_inputs.append([record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']])
        training_output.append(record['actual_occupancy'])

    classifier = MLPRegressor(solver='sgd', max_iter=1000, activation='relu', random_state=1, learning_rate_init=0.01,batch_size=len(training_inputs[0]),learning_rate='adaptive')
    classifier.fit(training_inputs, training_output)
    testing_inputs = []
    for record in testing_records:
        test_features = [record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']]
        testing_inputs.append(test_features)
    testing_predicted_output = classifier.predict(testing_inputs)
    k, error, precision_counter,error_class_0,error_class_1,error_class_2,size_class_0,size_class_1,size_class_2= 0, 0, 0, 0, 0, 0, 0, 0, 0
    for record in testing_records:
        error += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
        if occupancy_discretization.level(record['actual_occupancy'])==occupancy_discretization.level(testing_predicted_output[k]):
            precision_counter+=1

        if record['actual_occupancy']==0:
            error_class_0 += abs(occupancy_discretization.level(testing_predicted_output[k]) -occupancy_discretization.level(record['actual_occupancy']))
            size_class_0 +=1
        elif record['actual_occupancy'] == 1:
            error_class_1 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_1 += 1
        else:
            error_class_2 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_2 += 1
        k += 1

    return error / len(testing_records),testing_predicted_output,precision_counter/len(testing_records),error_class_0/size_class_0,error_class_1/size_class_1,error_class_2/size_class_2
class Logistic_regression:
    def __init__(self):
        self.weights=[1, 1, 1]
    def update_weights(self,new_weights:list):
         self.weights=new_weights
    def estimate_occupancy(self, training_records: Recording, testing_records: Recording):
        return self.Multinomial_logistic_regression(training_records,testing_records)[1]
    def get_weights(self):
        return self.weights
    def get_average_error_function(self,training_records:Recording,testing_records:Recording):
        def average_error_class1_class2(parameters: list=None):
            i=0
            if parameters is not None:
                self.weights = parameters
            size_class1=0
            size_class2=0
            error_class1=0
            error_class2=0
            for record in testing_records:
                if record['actual_occupancy']==1 :
                    error_class1 += abs(self.estimate_occupancy(training_records,testing_records)[i] - record['actual_occupancy'])
                    size_class1=size_class1+1
                elif record['actual_occupancy']==2 :
                    error_class2 += abs(self.estimate_occupancy(training_records,testing_records)[i] - record['actual_occupancy'])
                    size_class2=size_class2+1
                i=i+1
            average_error_class1=error_class1/size_class1
            average_error_class2 = error_class2 / size_class2
            return (average_error_class1)*(average_error_class2)
        return average_error_class1_class2
    def accept_test2(self, f_new, weights_new, f_old, weights_old):
        weight0= weights_new[0]
        weight1 = weights_new[1]
        weight2=weights_new[2]
        return bool((0 < weight0) and (0 < weight1)and(0 < weight2))
    
    def Multinomial_logistic_regression(self,training_records: Recording, testing_records: Recording,):
        training_inputs, training_output, training_errors = [], [], []
        occupancy_discretization = Discretization((0, 0), (0, 1.5), (1.5, 4))

        for record in training_records:
            #print("values I need ",record['actual_occupancy'],record['acoustic_pressure_dB'],record['estimated_occupancy_from_CO2'])
            training_inputs.append([record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']])
            training_output.append(occupancy_discretization.level(record['actual_occupancy']))
        weights_dict= { i : self.weights[i] for i in range(0, len(self.weights) ) }
        classifier = linear_model.LogisticRegression(penalty='l2',dual=False, tol=0.0001, C=0.7, fit_intercept=True,class_weight=weights_dict,random_state=1,multi_class='multinomial',solver='newton-cg',max_iter=1000)
        classifier.fit(training_inputs, training_output)
        testing_inputs = []
        for record in testing_records:
            test_features = [record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']]
            testing_inputs.append(test_features)
        testing_predicted_output = classifier.predict(testing_inputs)
        k, error, precision_counter,error_class_0,error_class_1,error_class_2,size_class_0,size_class_1,size_class_2= 0, 0, 0, 0, 0, 0, 0, 0, 0
        for record in testing_records:
            error += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            if occupancy_discretization.level(record['actual_occupancy'])==occupancy_discretization.level(testing_predicted_output[k]):
                precision_counter+=1

            if record['actual_occupancy']==0:
                error_class_0 += abs(occupancy_discretization.level(testing_predicted_output[k]) -occupancy_discretization.level(record['actual_occupancy']))
                size_class_0 +=1
            elif record['actual_occupancy'] == 1:
                error_class_1 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
                size_class_1 += 1
            else:
                error_class_2 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
                size_class_2 += 1
            k += 1

        return error / len(testing_records),testing_predicted_output,precision_counter/len(testing_records),error_class_0/size_class_0,error_class_1/size_class_1,error_class_2/size_class_2
    def  optimize_weights (self,training_records: Recording, testing_records: Recording):
        average_error = self.get_average_error_function(training_records,testing_records)()
        print('function average error ok')
        optimize_result = scipy.optimize.basinhopping(average_error, self.get_weights(),niter=10000, accept_test=self.accept_test2)

        return optimize_result.x


def KNN(training_records: Recording, testing_records: Recording):
    training_inputs, training_output, training_errors = [], [], []
    occupancy_discretization = Discretization((0, 0), (0, 1.5), (1.5, 4))

    for record in training_records:
        #print("values I need ",record['actual_occupancy'],record['acoustic_pressure_dB'],record['estimated_occupancy_from_CO2'])
        training_inputs.append([record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']])
        training_output.append(occupancy_discretization.level(record['actual_occupancy']))
    classifier = KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='auto',p=2)
    classifier.fit(training_inputs, training_output)
    testing_inputs = []
    for record in testing_records:
        test_features = [record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'], record['occupancy_from_power'], record['detected_motions']]
        testing_inputs.append(test_features)
    testing_predicted_output = classifier.predict(testing_inputs)
    k, error, precision_counter,error_class_0,error_class_1,error_class_2,size_class_0,size_class_1,size_class_2= 0, 0, 0, 0, 0, 0, 0, 0, 0
    for record in testing_records:
        error += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
        if occupancy_discretization.level(record['actual_occupancy'])==occupancy_discretization.level(testing_predicted_output[k]):
            precision_counter+=1

        if record['actual_occupancy']==0:
            error_class_0 += abs(occupancy_discretization.level(testing_predicted_output[k]) -occupancy_discretization.level(record['actual_occupancy']))
            size_class_0 +=1
        elif record['actual_occupancy'] == 1:
            error_class_1 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_1 += 1
        else:
            error_class_2 += abs(occupancy_discretization.level(testing_predicted_output[k]) - occupancy_discretization.level(record['actual_occupancy']))
            size_class_2 += 1
        k += 1

    return error / len(testing_records),testing_predicted_output,precision_counter/len(testing_records),error_class_0/size_class_0,error_class_1/size_class_1,error_class_2/size_class_2

        #function that plots the different performance indexes
def plot3 (error_to_plot,name_of_the_method,title):
    plt.figure()
    plt.bar(name_of_the_method,error_to_plot,color=['b', 'g', 'r', 'c', 'm', 'y', 'k'],align='center')
    plt.title(title)
    plt.show()

class AcousticCO2EstimationClassifier:

    def __init__(self, initial_parameters: list=[0.01, 1]):
        self.acoustic_pressure_dB_threshold = initial_parameters[0]
        self.estimated_occupancy_from_CO2_threshold = initial_parameters[1]
        self.acoustic_pressure_dB_threshold_bounds = (0, 1)
        self.estimated_occupancy_from_CO2_threshold_bounds = (0, 10)
        self.occupancy_discretization = Discretization((0, 0), (0, 1.5), (1.5, 4))

    @property
    def parameters(self):
        return [self.acoustic_pressure_dB_threshold, self.estimated_occupancy_from_CO2_threshold]

    @parameters.setter
    def parameters(self, updated_parameters: list):
        self.acoustic_pressure_dB_threshold = updated_parameters[0]
        self.estimated_occupancy_from_CO2_threshold = updated_parameters[1]

    def number_of_levels(self):
        return len(self.occupancy_discretization)

    def estimate_level(self, feature_variables: dict):
        if feature_variables['acoustic_pressure_dB'] <= self.acoustic_pressure_dB_threshold:
            return 0
        elif feature_variables['estimated_occupancy_from_CO2'] <= self.estimated_occupancy_from_CO2_threshold:
            return 1
        else:
            return 2

    def estimate_occupancy(self, feature_variables: dict):
        return self.occupancy_discretization.level_center(self.estimate_level(feature_variables))

    def accept_test(self, f_new, x_new, f_old, x_old):
        acoustic_pressure_dB_threshold = x_new[0]
        estimated_occupancy_from_CO2_threshold = x_new[1]
        return bool((0 < acoustic_pressure_dB_threshold < 1) and (0 < estimated_occupancy_from_CO2_threshold < 10))

    def get_level_accuracy_function(self, records):
        def level_accuracy(parameters: list=None):
            if parameters is not None:
                self.parameters = parameters
            number_of_properly_classified_records = 0
            for record in records:
                if self.estimate_level(record) == record['actual_occupancy']:
                    number_of_properly_classified_records += 1
            return number_of_properly_classified_records
        return level_accuracy

    def get_average_error_function(self, records):
        def average_error(parameters: list=None):
            if parameters is not None:
                self.parameters = parameters
            error = 0
            for record in records:
                error += abs(self.estimate_occupancy(record) - record['actual_occupancy'])
            return error / len(records)
        return average_error

    def plot(self, records, parameters: list=None):
        markers = {-1: 'xk', 0: 'ok', 1: '+k', 2: '*k', 3: '+r', 4: '*r'}
        if parameters is None:
            acoustic_pressure_dB_threshold = self.acoustic_pressure_dB_threshold
            estimated_occupancy_from_CO2_threshold = self.estimated_occupancy_from_CO2_threshold
        else:
            acoustic_pressure_dB_threshold = parameters[0]
            estimated_occupancy_from_CO2_threshold = parameters[1]
        plt.figure()
        for record in records:
            if record['actual_occupancy'] >= 0:
                plt.plot(record['acoustic_pressure_dB'], record['estimated_occupancy_from_CO2'], markers[self.occupancy_discretization.level(record['actual_occupancy'])])
                plt.plot([acoustic_pressure_dB_threshold, acoustic_pressure_dB_threshold], self.estimated_occupancy_from_CO2_threshold_bounds)
                plt.plot(self.acoustic_pressure_dB_threshold_bounds, [estimated_occupancy_from_CO2_threshold, estimated_occupancy_from_CO2_threshold])
                plt.xlabel('acoustic_pressure_dB')
                plt.ylabel('estimated_occupancy_from_CO2')


class InteractiveOccupancyEstimator():

    def __init__(self, seconds_in_sample: float=1800, starting_stringdatetime: str='4/05/2015 00:00:00', ending_stringdatetime: str='20/05/2015 00:00:00'):
        self.seconds_in_sample = seconds_in_sample
        data_container = h358data.H358DataContainer(sample_time=seconds_in_sample, starting_stringdatetime=starting_stringdatetime, ending_stringdatetime=ending_stringdatetime)
        self.all_records = Recording()
        self.all_records.add_serie('epochtime', data_container.get_variable('epochtime'))
        self.all_records.add_serie('office_CO2_concentration', data_container.get_variable('office_CO2_concentration'))
        self.all_records.add_serie('corridor_CO2_concentration', data_container.get_variable('corridor_CO2_concentration'))
        self.all_records.add_serie('door_opening', data_container.get_variable('door_opening'))
        self.all_records.add_serie('window_opening', data_container.get_variable('window_opening'))
        self.all_records.add_serie('detected_motions', data_container.get_variable('detected_motions'))
        self.all_records.add_serie('occupancy_from_power', data_container.get_variable('occupancy'))
        self.all_records.add_serie('acoustic_pressure_dB', data_container.get_variable('acoustic_pressure_dB'))
        self.all_records.add_serie('actual_occupancy', data_container.get_variable('actual_occupation'))
        file = open('best_parameters.sav', 'rb')
        pset = pickle.load(file)
        file.close()
        occupancy_estimator_from_CO2 = generate_occupancy_estimator_from_CO2(pset)
        estimated_occupancies_from_CO2 = occupancy_estimator_from_CO2(self.all_records.serie('epochtime'), self.all_records.serie('office_CO2_concentration'), self.all_records.serie('corridor_CO2_concentration'), self.all_records.serie('door_opening'), self.all_records.serie('window_opening'))
        self.all_records.add_serie('estimated_occupancy_from_CO2', estimated_occupancies_from_CO2)
        self.ask_records = Recording()
        self.recording_samples = list()
        self.classifier = AcousticCO2EstimationClassifier()
        self.class_counters = [0 for i in range(self.classifier.number_of_levels())]
        self.classifier.plot(self.all_records, [0.012, 0.74])
        # self.epsilon=spread_rate(self.all_records,None)

            #integration of spreadrate as an internal function of interactive estimator

    def spread_rate(self,raw_points_recording:Recording, search_space_ranges: 'list of ranges' = None):
        """
        compute a spread rate corresponding to the average minimum distance seperating each point from the closest point
        :param raw_points: list of points; by default list of vector of values, but opposite dimension is accepted
        :param search_space_ranges: list of 2-dimensional intervalles represented the min and max values for coordinate of a point
                if None, values are not changed if there lies between 0 and 1 or normalized by minimum and maximum values
        :return: the spread rate. 1 correspond to a perfect spread and 0 to several points at the same position
        """
        raw_points = []
        for record in raw_points_recording:
            raw_points.append([record['door_opening'], record['estimated_occupancy_from_CO2'], record['acoustic_pressure_dB'],record['occupancy_from_power'], record['detected_motions']])

        if len(raw_points) >= len(raw_points[0]):  # list of vector of d-dimensional values
            dimension = len(raw_points[0])
        else:  # d lists of values
            dimension = len(raw_points)
            points = list()
            for i in range(len(raw_points[0])):
                point = list()
                for j in range(len(raw_points)):
                    point.append(raw_points[j][i])
                points.append(point)
            raw_points = points
        if search_space_ranges is None:
            search_space_ranges = [[0, 1] for i in range(dimension)]
            for i in range(len(raw_points)):
                for j in range(dimension):
                    if raw_points[i][j] < search_space_ranges[j][0]:
                        search_space_ranges[j][0] = raw_points[i][j]
                    elif raw_points[i][j] > search_space_ranges[j][1]:
                        search_space_ranges[j][1] = raw_points[i][j]
        points = list()
        for raw_point in raw_points:
            point = list()
            for i in range(dimension):
                point.append((raw_point[i] - search_space_ranges[i][0]) / (
                            search_space_ranges[i][1] - search_space_ranges[i][0]))
            points.append(point)
        min_dists = list()
        for i in range(len(points)):
            min_dist = list()
            for j in range(len(points)):
                if i != j:
                    min_dist.append(max([abs(points[i][k] - points[j][k]) for k in
                                         range(dimension)]))  # Tchebychev distance calculation
            min_dists.append(min(min_dist))
        score = sum(min_dists) / len(min_dists) * (len(points) ** (1 / dimension) - 1)
        return score
    # epsilon=spread_rate(self.all_records,None)


    def interactive_learn(self,name_algorithm: str='decision_tree'):
        self.ask_records = Recording()
        self.recording_samples = list()
        self.class_counters = [0 for i in range(self.classifier.number_of_levels())]
        self.recording_samples.append(0)
        self.ask_records.append(self.all_records[0])
        self.class_counters[self.classifier.estimate_level(self.ask_records[-1])] += 1
        average_testing_error=1000
        self.classifier2 = Logistic_regression()
        self.name_algorithm=name_algorithm
        training_errors = None

        #choosing the function corresponding to the algorithm wanted

        if self.name_algorithm=='decision_tree':
            self.algorithm=decision_tree
        elif self.name_algorithm=='support_vector_machine':
            self.algorithm = decision_tree
        elif self.name_algorithm=='naive_bayes':
            self.algorithm = naive_bayes
        elif self.name_algorithm=='random_forest':
            self.algorithm=random_forest
        elif self.name_algorithm=='Neural_Network':
            self.algorithm=Neural_Network
        elif self.name_algorithm=='KNN':
            self.algorithm=KNN
        else:
            print("the algorithm is logistic regression")

            #integration of spreadrate for building ask records and calculation of the different errors
        for k in range(1, len(self.all_records)):
            if average_testing_error>0.23:
                ask = dict(self.all_records[k])
                records=self.ask_records
                self.recording_samples.append(k)
                spread_rate_records1 = self.spread_rate(raw_points_recording=records, search_space_ranges=None)
                records.append(ask)
                spread_rate_records2 = self.spread_rate(raw_points_recording=records, search_space_ranges=None)
                if spread_rate_records2 > spread_rate_records1:
                    self.ask_records.append(ask)

                average_error = self.classifier.get_average_error_function(self.ask_records)
                optimize_result = scipy.optimize.basinhopping(average_error, self.classifier.parameters,
                                                              niter=10000, accept_test=self.classifier.accept_test)
                self.classifier.parameters = optimize_result.x
                self.class_counters[self.classifier.estimate_level(ask)] += 1
                average_error = self.classifier.get_average_error_function(self.all_records)
                class_accuracy = self.classifier.get_level_accuracy_function(self.all_records)

                average_testing_error, training_errors,precision,average_error_class0,average_error_class1,average_error_class2 = self.algorithm(self.ask_records, self.all_records)

                print('> ASK#', len(self.ask_records), ' thresholds: ', optimize_result.x,
                      ', properly classified: ', class_accuracy(), ' / ', len(self.all_records),
                      ', estimator error: ', average_error(optimize_result.x),'  average testing error of  ', name_algorithm,
                      average_testing_error, sep='')

            else:
                print('average_testing_error  ', average_testing_error)
                break

            #self.classifier.plot(self.ask_records)

        return average_testing_error, average_error(optimize_result.x)  # decision tree, parameterized estimator

    def get_day_ask_frequency(self):
        samples_per_day = 24 * 3600 / self.seconds_in_sample
        day_ask_frequency = dict()
        for recording_sample in self.recording_samples:
            day = int(recording_sample // samples_per_day)
            if day not in day_ask_frequency:
                day_ask_frequency[day] = 1
            else:
                day_ask_frequency[day] += 1
        number_of_asks_per_day = [0 for i in range(max(day_ask_frequency) + 1)]
        for day in day_ask_frequency:
            number_of_asks_per_day[day] = day_ask_frequency[day]
        return number_of_asks_per_day

    def get_class_counter(self):
        return self.class_counters


class Test(unittest.TestCase):
    def plot2(self,support_vector_machine,training_records:Recording,testing_records:Recording):
        testing_records_occupancy=[]
        testing_predicted_output=support_vector_machine(training_records, testing_records)

        for record in testing_records:
            testing_records_occupancy.append(record['actual_occupancy'])
        bins = np.linspace(-10, 10, 30)
        plt.hist([testing_records_occupancy, testing_predicted_output], bins, label=['testing_records_occupancy', 'testing_predicted_output'])
        plt.legend(loc='upper right')
        plt.show()

    def test_estimator(self):

        average_error_decision_tree_100,accuracy_decision_tree_100,average_error_class0_decision_tree_100,average_error_class1_decision_tree_100,average_error_class2_decision_tree_100=[],[],[],[],[]
        average_error_support_vector_machine_100,accuracy_support_vector_machine_100,average_error_class0_support_vector_machine_100,average_error_class1_support_vector_machine_100,average_error_class2_support_vector_machine_100=[],[],[],[],[]
        average_error_Naive_Bayes_100,accuracy_Naive_Bayes_100,average_error_class0_Naive_Bayes_100,average_error_class1_Naive_Bayes_100,average_error_class2_Naive_Bayes_100=[],[],[],[],[]
        average_error_Random_forest_100,accuracy_Random_forest_100,average_error_class0_Random_forest_100,average_error_class1_Random_forest_100,average_error_class2_Random_forest_100=[],[],[],[],[]
        average_error_Neural_network_100,accuracy_Neural_network_100,average_error_class0_Neural_network_100,average_error_class1_Neural_network_100,average_error_class2_Neural_network_100=[],[],[],[],[]
        average_error_Logistic_regression_100,accuracy_Logistic_regression_100,average_error_class0_Logistic_regression_100,average_error_class1_Logistic_regression_100,average_error_class2_Logistic_regression_100=[],[],[],[],[]
        average_error_KNN_100,accuracy_KNN_100,average_error_class0_KNN_100,average_error_class1_KNN_100,average_error_class2_KNN_100=[],[],[],[],[]


        for i in range(1):
            interactive_occupancy_estimator = InteractiveOccupancyEstimator(1800, '4/05/2015 00:30:00', '14/05/2015 00:30:00')
            interactive_occupancy_estimator.interactive_learn(name_algorithm='decision_tree')

            average_error_decision_tree_100.append(decision_tree(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[0])
            accuracy_decision_tree_100.append(decision_tree(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[2])
            average_error_class0_decision_tree_100.append(decision_tree(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[3])
            average_error_class1_decision_tree_100.append(decision_tree(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[4])
            average_error_class2_decision_tree_100.append(decision_tree(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[5])

            # average_error_support_vector_machine_100.append(support_vector_machine(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[0])
            # accuracy_support_vector_machine_100.append(support_vector_machine(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[2])
            # average_error_class0_support_vector_machine_100.append(support_vector_machine(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[3])
            # average_error_class1_support_vector_machine_100.append(support_vector_machine(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[4])
            # average_error_class2_support_vector_machine_100.append(support_vector_machine(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[5])

            average_error_Naive_Bayes_100.append(naive_bayes(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[0])
            accuracy_Naive_Bayes_100.append(naive_bayes(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[2])
            average_error_class0_Naive_Bayes_100.append(naive_bayes(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[3])
            average_error_class1_Naive_Bayes_100.append(naive_bayes(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[4])
            average_error_class2_Naive_Bayes_100.append(naive_bayes(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[5])

            average_error_Random_forest_100.append(random_forest(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[0])
            accuracy_Random_forest_100.append(random_forest(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[2])
            average_error_class0_Random_forest_100.append(random_forest(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[3])
            average_error_class1_Random_forest_100.append(random_forest(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[4])
            average_error_class2_Random_forest_100.append(random_forest(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[5])

            average_error_Neural_network_100.append(Neural_Network(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[0])
            accuracy_Neural_network_100.append(Neural_Network(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[2])
            average_error_class0_Neural_network_100.append(Neural_Network(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[3])
            average_error_class1_Neural_network_100.append(Neural_Network(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[4])
            average_error_class2_Neural_network_100.append(Neural_Network(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[5])

# This is where i optimize the weights that are going to be used next when i call the logistic regression afterwards
            print('ask records  ',len(interactive_occupancy_estimator.ask_records))
# My error is located here in the function optimize weights
            optimized_weights=interactive_occupancy_estimator.classifier2.optimize_weights(interactive_occupancy_estimator.ask_records,interactive_occupancy_estimator.all_records)


            print('optimal weights=   ', optimized_weights)
            interactive_occupancy_estimator.classifier2.update_weights(optimized_weights)

            average_error_Logistic_regression_100.append(interactive_occupancy_estimator.classifier2.Multinomial_logistic_regression(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[0])
            accuracy_Logistic_regression_100.append(interactive_occupancy_estimator.classifier2.Multinomial_logistic_regression(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[2])
            average_error_class0_Logistic_regression_100.append(interactive_occupancy_estimator.classifier2.Multinomial_logistic_regression(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[3])
            average_error_class1_Logistic_regression_100.append(interactive_occupancy_estimator.classifier2.Multinomial_logistic_regression(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[4])
            average_error_class2_Logistic_regression_100.append(interactive_occupancy_estimator.classifier2.Multinomial_logistic_regression(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[5])

            average_error_KNN_100.append(KNN(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[0])
            accuracy_KNN_100.append(KNN(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[2])
            average_error_class0_KNN_100.append(KNN(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[3])
            average_error_class1_KNN_100.append(KNN(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[4])
            average_error_class2_KNN_100.append(KNN(interactive_occupancy_estimator.ask_records, interactive_occupancy_estimator.all_records)[5])

        # final_error_decision_tree_100,final_accuracy_decision_tree_100,final_error_class0_decision_tree_100,final_error_class1_decision_tree_100,final_error_class2_decision_tree_100=0,0,0,0,0
        # final_error_support_vector_machine_100,final_accuracy_support_vector_machine_100,final_error_class0_support_vector_machine_100,final_error_class1_support_vector_machine_100,final_error_class2_support_vector_machine_100=0,0,0,0,0
        # final_error_Naive_Bayes_100,final_accuracy_Naive_Bayes_100,final_error_class0_Naive_Bayes_100,final_error_class1_Naive_Bayes_100,final_error_class2_Naive_Bayes_100=0,0,0,0,0
        # final_error_Random_forest_100,final_accuracy_Random_forest_100,final_error_class0_Random_forest_100,final_error_class1_Random_forest_100,final_error_class2_Random_forest_100=0,0,0,0,0
        # final_error_Neural_network_100,final_accuracy_Neural_network_100,final_error_class0_Neural_network_100,final_error_class1_Neural_network_100,final_error_class2_Neural_network_100=0,0,0,0,0
        # final_error_Logistic_regression_100,final_accuracy_Logistic_regression_100,final_error_class0_Logistic_regression_100,final_error_class1_Logistic_regression_100,final_error_class2_Logistic_regression_100=0,0,0,0,0
        # final_error_KNN_100,final_accuracy_KNN_100,final_error_class0_KNN_100,final_error_class1_KNN_100,final_error_class2_KNN_100=0,0,0,0,0

        print(average_error_decision_tree_100)

        final_error_decision_tree_100=np.average(average_error_decision_tree_100)
        final_accuracy_decision_tree_100=np.average(accuracy_decision_tree_100)
        final_error_class0_decision_tree_100=np.average(average_error_class0_decision_tree_100)
        final_error_class1_decision_tree_100 = np.average(average_error_class1_decision_tree_100)
        final_error_class2_decision_tree_100 = np.average(average_error_class2_decision_tree_100)

        final_error_support_vector_machine_100=np.average(average_error_support_vector_machine_100)
        final_accuracy_support_vector_machine_100=np.average(accuracy_support_vector_machine_100)
        final_error_class0_support_vector_machine_100=np.average(average_error_class0_support_vector_machine_100)
        final_error_class1_support_vector_machine_100 = np.average(average_error_class1_support_vector_machine_100)
        final_error_class2_support_vector_machine_100 = np.average(average_error_class2_support_vector_machine_100)

        final_error_Naive_Bayes_100=np.average(average_error_Naive_Bayes_100)
        final_accuracy_Naive_Bayes_100=np.average(accuracy_Naive_Bayes_100)
        final_error_class0_Naive_Bayes_100=np.average(average_error_class0_Naive_Bayes_100)
        final_error_class1_Naive_Bayes_100 = np.average(average_error_class1_Naive_Bayes_100)
        final_error_class2_Naive_Bayes_100 = np.average(average_error_class2_Naive_Bayes_100)

        final_error_Random_forest_100=np.average(average_error_Random_forest_100)
        final_accuracy_Random_forest_100=np.average(accuracy_Random_forest_100)
        final_error_class0_Random_forest_100=np.average(average_error_class0_Random_forest_100)
        final_error_class1_Random_forest_100 = np.average(average_error_class1_Random_forest_100)
        final_error_class2_Random_forest_100 = np.average(average_error_class2_Random_forest_100)

        final_error_Neural_network_100=np.average(average_error_Neural_network_100)
        final_accuracy_Neural_network_100=np.average(accuracy_Neural_network_100)
        final_error_class0_Neural_network_100=np.average(average_error_class0_Neural_network_100)
        final_error_class1_Neural_network_100 = np.average(average_error_class1_Neural_network_100)
        final_error_class2_Neural_network_100 = np.average(average_error_class2_Neural_network_100)

        final_error_Logistic_regression_100=np.average(average_error_Logistic_regression_100)
        final_accuracy_Logistic_regression_100=np.average(accuracy_Logistic_regression_100)
        final_error_class0_Logistic_regression_100=np.average(average_error_class0_Logistic_regression_100)
        final_error_class1_Logistic_regression_100 = np.average(average_error_class1_Logistic_regression_100)
        final_error_class2_Logistic_regression_100 = np.average(average_error_class2_Logistic_regression_100)

        final_error_KNN_100=np.average(average_error_KNN_100)
        final_accuracy_KNN_100=np.average(accuracy_KNN_100)
        final_error_class0_KNN_100=np.average(average_error_class0_KNN_100)
        final_error_class1_KNN_100 = np.average(average_error_class1_KNN_100)
        final_error_class2_KNN_100 = np.average(average_error_class2_KNN_100)

        error_to_plot1 = [final_error_decision_tree_100,final_error_support_vector_machine_100,final_error_Naive_Bayes_100,final_error_Random_forest_100,final_error_Neural_network_100,final_error_Logistic_regression_100,final_error_KNN_100]
        error_to_plot2 = [final_accuracy_decision_tree_100,final_accuracy_support_vector_machine_100,final_accuracy_Naive_Bayes_100,final_accuracy_Random_forest_100,final_accuracy_Neural_network_100,final_accuracy_Logistic_regression_100,final_accuracy_KNN_100]
        error_to_plot3 = [final_error_class0_decision_tree_100,final_error_class0_support_vector_machine_100,final_error_class0_Naive_Bayes_100,final_error_class0_Random_forest_100,final_error_class0_Neural_network_100,final_error_class0_Logistic_regression_100,final_error_class0_KNN_100]
        error_to_plot4 = [final_error_class1_decision_tree_100,final_error_class1_support_vector_machine_100,final_error_class1_Naive_Bayes_100,final_error_class1_Random_forest_100,final_error_class1_Neural_network_100,final_error_class1_Logistic_regression_100,final_error_class1_KNN_100]
        error_to_plot5 = [final_error_class2_decision_tree_100,final_error_class2_support_vector_machine_100,final_error_class2_Naive_Bayes_100,final_error_class2_Random_forest_100,final_error_class2_Neural_network_100,final_error_class2_Logistic_regression_100,final_error_class2_KNN_100]

        print(error_to_plot1)
        name_of_the_method=["decision\ntree","support\nvector\nmachine","Naive\nBayes","random\nforest","neural\nnetwork","logistic\nregression","KNN"]
        plot3(error_to_plot1, name_of_the_method, "Average error")
        plot3(error_to_plot2, name_of_the_method, "Accuracy")
        plot3(error_to_plot3, name_of_the_method, "Average error from class 0")
        plot3(error_to_plot4, name_of_the_method, "Average error from class 1")
        plot3(error_to_plot5, name_of_the_method, "Average error from class 2")
if __name__ == '__main__':
    unittest.main()
