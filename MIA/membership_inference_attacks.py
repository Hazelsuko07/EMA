import numpy as np
import math

class black_box_benchmarks(object):
    
    def __init__(self, shadow_train_performance, shadow_test_performance, 
                 target_train_performance, target_test_performance, x_target, y_target,x_shadow, y_shadow,num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''
        self.num_classes = num_classes
        self.x_target = x_target
        self.y_target = y_target
        self.x_shadow = x_shadow
        self.y_shadow = y_shadow

        # confidence vector
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance      
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        
        # label matching (label-only)
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        
        # entropy
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)
        
        # modified entropy
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
        
    
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)
    
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)
    
    def _thre_setting(self, tr_values, te_values):
        """Decide the threshold for membership inference attacks using the shadow data

        Args:
            tr_values ([float]): a list of values for shadow train set
            te_values ([float]): a list of values for shadow test set

        Returns:
            float: the threshold
        """
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre
    
    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        return mem_inf_acc, t_tr_acc, t_te_acc
    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)
        mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
        return mem_inf_acc
    
    # def _mem_inf_NN(self, args):
    #     import configparser
    #     import imp
    #     import keras

    #     config = configparser.ConfigParser()
    #     config.read('config.ini')
    #     result_folder=config[args.dataset]["result_folder"]
    #     vanilla_epochs=int(config[args.dataset]["vanilla_epochs"])
    #     network_architecture=str(config[args.dataset]["network_architecture"])
    #     fccnet=imp.load_source(str(config[args.dataset]["network_name"]),network_architecture)
    #     # 1. load the shadow model according to args
    #     if args.adaptive: ## TODO: generate ada model, and load as expected
    #         npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_attack_shallow_model_adv_ada.npz".format(vanilla_epochs), allow_pickle=True)
    #     else:
    #         npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_attack_shallow_model_adv1.npz".format(vanilla_epochs), allow_pickle=True,encoding = 'latin1')
    #     weights=npzdata['x']
        
    #     input_shape=self.x_shadow.shape[1:]
    #     model=fccnet.model_vanilla(input_shape=input_shape,labels_dim=self.num_classes)
    #     model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
    #     model.set_weights(weights)
    #     # f_train=model.predict(x_train)
    #     # del model
    #     # f_train=np.sort(f_train,axis=1)
    #     # f_evaluate_defense=np.sort(f_evaluate_defense,axis=1)
    #     # f_evaluate_origin=np.sort(f_evaluate_origin,axis=1)
    #     ## TODO: 
    #     # 2. calculate the performance
    #     print('To be implemented')
    #     return 0

    
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[], args=None):
        best_acc = 0
        res = []
        if (all_methods) or ('correctness' in benchmark_methods):
            acc_corr, train_acc, test_acc = self._mem_inf_via_corr()
            res.extend([train_acc, test_acc, acc_corr])
            if acc_corr > best_acc:
                best_acc = acc_corr
        if (all_methods) or ('confidence' in benchmark_methods):
            acc_conf = self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
            res.append(acc_conf)
            if acc_conf > best_acc:
                best_acc = acc_conf
        if (all_methods) or ('entropy' in benchmark_methods):
            acc_ent = self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
            res.append(acc_ent)
            if acc_ent > best_acc:
                best_acc = acc_ent
        if (all_methods) or ('modified entropy' in benchmark_methods):
            acc_ment = self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)
            res.append(acc_ment)
            if acc_ment > best_acc:
                best_acc = acc_ment
        ## TODO: add API for NN attacks
        # if (all_methods) or ('NN' in benchmark_methods):
        #     print('To be implemented')
        #     acc_NN = self._mem_inf_NN(args)
        
        print(f'Best attack acc: {best_acc}')
        res.append(best_acc)
        return res