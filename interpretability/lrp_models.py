#-------------------------- LRP ---------------------------------------#

import torch
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule, IdentityRule
import sslearning 

'''some class to initalise LRP rules'''
 #to-do:
    #1. probably a smarter way of making these classes more customizable 

def _get_MTL_classifer_layers(model, MTL_classifer_layer_names=['aot_h', 'permute_h', 'permute_h', 'scale_h']):
    classifer_layers=[]
    for MTL_classifer_layer in MTL_classifer_layer_names:
        class_layer=_get_model_layers(model._modules[MTL_classifer_layer])
        for layer in class_layer:
            classifer_layers.append(layer)
    return classifer_layers

class LRPEpsilonBenchmark(object):
    def __init__(self,verbose=True):
        self.verbose=verbose

    def __call__(self, model):
            
        components=_get_model_layers(model)
        rules=[]
        for c, layer in enumerate(components):
            setattr(layer, 'rule', EpsilonRule());rule='LRP-e'
            rules.append([c, rule]) 

        return model, rules  

    def __reset__(self, model):
        model, _=_reset_lrp_rules(model)
        return model

class LRPEpsilonSSL(object):
   
    def __init__(self,verbose=True, epsilon=1e-9):
        self.verbose=verbose
        self.epsilon=epsilon

    def __call__(self, model):
            
        components=_get_model_layers(model)
        number_components=len(components)
       
        feature_extraction_blocks = list(model._modules['feature_extractor'])
        classifer_layers=_get_model_layers(model._modules['classifier'])

        '''set feature exraction block rules'''
        rules=[]
        for b, block in enumerate(feature_extraction_blocks):

            rule='default'
            if isinstance(block, torch.nn.modules.container.Sequential):
                for l, layer in enumerate(block):
                    #set convolutional layer rules 
                    if isinstance(layer, torch.nn.modules.conv.Conv1d):

                        if b<2:
                            setattr(layer, 'rule', EpsilonRule(epsilon=self.epsilon));rule='LRP-e'
                        elif (2 <= b <= 5): 
                            setattr(layer, 'rule', EpsilonRule(epsilon=self.epsilon));rule='LRP-e'
                        elif b >= 5:
                            setattr(layer, 'rule', EpsilonRule(epsilon=self.epsilon));rule='LRP-e'
        
                    elif isinstance(layer, sslearning.models.accNet.ResBlock): 
                        # set residual block layer rules
                        res_layers=_get_model_layers(layer)

                        for res_layer in res_layers:
                            if isinstance(res_layer, torch.nn.modules.batchnorm.BatchNorm1d):
                                setattr(res_layer, 'rule', EpsilonRule(epsilon=self.epsilon));rule='LRP-e'
                            if isinstance(res_layer, torch.nn.modules.conv.Conv1d):
                                setattr(res_layer, 'rule', EpsilonRule(epsilon=self.epsilon));rule='LRP-e'

                    elif isinstance(layer, torch.nn.modules.batchnorm.BatchNorm1d):
                        setattr(layer, 'rule', EpsilonRule(epsilon=self.epsilon));rule='LRP-e'

                    elif isinstance(layer, sslearning.models.accNet.Downsample):
                            setattr(layer, 'rule', EpsilonRule(epsilon=self.epsilon));rule='LRP-e' 

                    else: rule='none'
            else: rule='none'

            #append rules
            rules.append([b, l, rule]) 
        
        for c, class_layer in enumerate(classifer_layers):
            if isinstance(class_layer, torch.nn.Linear):
                setattr(class_layer, 'rule', Alpha1_Beta0_Rule());rule='LRP-0' # LRP-0:upper-layers (classification)
            rules.append([b+c, 0, rule]) 

        if self.verbose:
            #run on the model to verify the model has been changed
            _print_lrp_rules(model)

        return model, rules  

    def __reset__(self, model):
        model, _=_reset_lrp_rules(model)
        return model


class LRPCompositeSSL(object):
   
    def __init__(self,verbose=True, epsilon=1e-9):
        self.verbose=verbose
        self.epsilon=epsilon
    def __call__(self, model):
            
        components=_get_model_layers(model)
        number_components=len(components)
       
        feature_extraction_blocks = list(model._modules['feature_extractor'])
        classifer_layers=_get_model_layers(model._modules['classifier'])
        
        #set feature exraction block rules#
        rules=[];rules.append(['role', 'block', 'layer', 'rule'])
        for b, block in enumerate(feature_extraction_blocks):

            rule='default'
            if isinstance(block, torch.nn.modules.container.Sequential):
                for l, layer in enumerate(block):
                    #set convolutional layer rules 
                    if isinstance(layer, torch.nn.modules.conv.Conv1d):

                        if b<4:
                            setattr(layer, 'rule', GammaRule());rule='LRP-y'
                        elif (4 <= b <= 15): 
                            setattr(layer, 'rule', EpsilonRule(1e-3));rule='LRP-AB'
                        elif b >= 15:
                            setattr(layer, 'rule', EpsilonRule(epsilon=1e-9));rule='LRP-e'
        
                    elif isinstance(layer, sslearning.models.accNet.ResBlock): 
                        # set residual block layer rules
                        res_layers=_get_model_layers(layer)

                        for res_layer in res_layers:
                            if isinstance(res_layer, torch.nn.modules.batchnorm.BatchNorm1d):
                                setattr(res_layer, 'rule', EpsilonRule());rule='LRP-e'
                            if isinstance(res_layer, torch.nn.modules.conv.Conv1d):
                                setattr(res_layer, 'rule', EpsilonRule(epsilon=10));rule='LRP-e'

                    elif isinstance(layer, torch.nn.modules.batchnorm.BatchNorm1d):
                        setattr(layer, 'rule', EpsilonRule());rule='LRP-e'

                    elif isinstance(layer, sslearning.models.accNet.Downsample):
                            setattr(layer, 'rule', EpsilonRule());rule='LRP-e' 

                    else: rule='none'

                    #append rules
                    rules.append(['feature_extractor', b, l, rule]) 

            else: 
                rule='none'
                #append rules
                rules.append([b, l, rule]) 
        b+=1    
        for c, class_layer in enumerate(classifer_layers):
            if isinstance(class_layer, torch.nn.Linear):
                setattr(class_layer, 'rule', EpsilonRule(epsilon=0));rule='LRP-0' # LRP-0:upper-layers (classification)
            rules.append(['classifier', b, c, rule]) 

        if self.verbose:
            #run on the model to verify the model has been changed
            _print_lrp_rules(model)

        return model, rules  

    def __reset__(self, model):
        model, _=_reset_lrp_rules(model)
        return model


class LRPCompositeSSL_AlphaBeta(object):
   
    def __init__(self,verbose=True, epsilon=100):
        self.verbose=verbose
        self.epsilon=epsilon
    def __call__(self, model):
            
        components=_get_model_layers(model)
        number_components=len(components)
       
        feature_extraction_blocks = list(model._modules['feature_extractor'])
        classifer_layers=_get_model_layers(model._modules['classifier'])
        
        #set feature exraction block rules
        rules=[];rules.append(['role', 'block', 'layer', 'rule'])
        for b, block in enumerate(feature_extraction_blocks):

            rule='default'
            if isinstance(block, torch.nn.modules.container.Sequential):
                for l, layer in enumerate(block):
                    #set convolutional layer rules 
                    if isinstance(layer, torch.nn.modules.conv.Conv1d):

                        if b<3:
                            setattr(layer, 'rule', Alpha1_Beta0_Rule());rule='LRP-y'
                        elif (3 <= b <= 6): 
                            setattr(layer, 'rule', EpsilonRule(epsilon=1e-3));rule='LRP-e'
                        elif b >= 16:
                            setattr(layer, 'rule', EpsilonRule(epsilon=1e-9));rule='LRP-e'
        
                    elif isinstance(layer, sslearning.models.accNet.ResBlock): 
                        # set residual block layer rules
                        res_layers=_get_model_layers(layer)

                        for res_layer in res_layers:
                            if isinstance(res_layer, torch.nn.modules.batchnorm.BatchNorm1d):
                                setattr(res_layer, 'rule', EpsilonRule());rule='LRP-e'
                            if isinstance(res_layer, torch.nn.modules.conv.Conv1d):
                                setattr(res_layer, 'rule', EpsilonRule(epsilon=2));rule='LRP-e'

                    elif isinstance(layer, torch.nn.modules.batchnorm.BatchNorm1d):
                        setattr(layer, 'rule', EpsilonRule());rule='LRP-e'

                    elif isinstance(layer, sslearning.models.accNet.Downsample):
                            setattr(layer, 'rule', EpsilonRule());rule='LRP-e' 

                    else: rule='none'

                    #append rules
                    rules.append(['feature_extractor', b, l, rule]) 

            else: 
                rule='none'
                #append rules
                rules.append([b, l, rule]) 
        b+=1    
        for c, class_layer in enumerate(classifer_layers):
            if isinstance(class_layer, torch.nn.Linear):
                setattr(class_layer, 'rule', EpsilonRule(epsilon=0));rule='LRP-0' # LRP-0:upper-layers (classification)
            rules.append(['classifier', b, c, rule]) 

        if self.verbose:
            #run on the model to verify the model has been changed
            _print_lrp_rules(model)

        return model, rules  

    def __reset__(self, model):
        model, _=_reset_lrp_rules(model)
        return model

class LRPAlpha1Beta0SSL(object):
   
    def __init__(self,verbose=True, epsilon=1e-9):
        self.verbose=verbose
        self.epsilon=epsilon

    def __call__(self, model):
            
        components=_get_model_layers(model)
        number_components=len(components)
       
        feature_extraction_blocks = list(model._modules['feature_extractor'])
        classifer_layers=_get_model_layers(model._modules['classifier'])

        '''set feature exraction block rules'''
        rules=[]
        for b, block in enumerate(feature_extraction_blocks):

            rule='default'
            if isinstance(block, torch.nn.modules.container.Sequential):
                for l, layer in enumerate(block):
                    #set convolutional layer rules 
                    if isinstance(layer, torch.nn.modules.conv.Conv1d):

                        if b<2:
                            setattr(layer, 'rule', Alpha1_Beta0_Rule());rule='LRP-AB'
                        elif (2 <= b <= 5): 
                            setattr(layer, 'rule', Alpha1_Beta0_Rule());rule='LRP-AB'
                        elif b >= 5:
                            setattr(layer, 'rule', Alpha1_Beta0_Rule());rule='LRP-AB'
        
                    elif isinstance(layer, sslearning.models.accNet.ResBlock): 
                        # set residual block layer rules
                        res_layers=_get_model_layers(layer)

                        for res_layer in res_layers:
                            if isinstance(res_layer, torch.nn.modules.batchnorm.BatchNorm1d):
                                setattr(res_layer, 'rule', EpsilonRule());rule='LRP-e'
                            if isinstance(res_layer, torch.nn.modules.conv.Conv1d):
                                setattr(res_layer, 'rule', Alpha1_Beta0_Rule());rule='LRP-e'

                    elif isinstance(layer, torch.nn.modules.batchnorm.BatchNorm1d):
                        setattr(layer, 'rule', EpsilonRule());rule='LRP-e'

                    elif isinstance(layer, sslearning.models.accNet.Downsample):
                            setattr(layer, 'rule', EpsilonRule());rule='LRP-e' 

                    else: rule='none'
            else: rule='none'

            #append rules
            rules.append([b, l, rule]) 
        '''
        for c, class_layer in enumerate(classifer_layers):
            if isinstance(class_layer, torch.nn.Linear):
                setattr(class_layer, 'rule', EpsilonRule(epsilon=0));rule='LRP-0' # LRP-0:upper-layers (classification)
            rules.append([b+c, 0, rule]) 
        '''
        if self.verbose:
            #run on the model to verify the model has been changed
            _print_lrp_rules(model)

        return model, rules  

    def __reset__(self, model):
        model, _=_reset_lrp_rules(model)
        return model

#-------------------------- Utils -------------------------------------#
def _get_model_layers(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(_get_model_layers(child))
            except TypeError:
                flatt_children.append(_get_model_layers(child))
    return flatt_children

def _reset_lrp_rules(model):
    #run this code to re-set all the layers to default epsilon-rule
    layers=_get_model_layers(model)
    number_layers=len(layers)
    rules=[]
    for l in range(0,number_layers)[::-1]:
        setattr(layers[l], 'rule', EpsilonRule())
        rules.append([l, 'rule', layers[l]])    
    return model, rules

def _print_lrp_rules(model):
    layers=_get_model_layers(model)
    number_layers=len(layers)
    for l in range(0,number_layers)[::-1]:
        try:
            rule=getattr(layers[l], 'rule')  
        except:
            rule='default'

        print('layer {:}| rule: {:} \n    structure: {:}\n----------------------------------------------------------------------------'.format(l, rule, layers[l]))
   