#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 19:37:28 2018

@author: lost
"""

import coremltools
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
 
output_labels = ['dress','shirt','jeans']
scale = 1/255.
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    coreml_model = coremltools.converters.keras.convert('new2.h5',
                                                       input_names='image',
                                                       image_input_names='image',
                                                       output_names='output',
                                                       class_labels=output_labels,
                                                       image_scale=scale)
     
    coreml_model.author = 'Dogukan Demirci'
    coreml_model.license = 'MIT'
    coreml_model.short_description = 'Model to classify fashion image'
     
    coreml_model.input_description['image'] = 'Grayscale image of fashion image'
    coreml_model.output_description['output'] = 'Predicted fashion'
     
    coreml_model.save('fashionCNN.mlmodel')
