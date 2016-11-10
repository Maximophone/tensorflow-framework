import tensorflow as tf
import inspect
import os
import json

_globals = {
    'master':None
}

def hash_function(f):
    source_lines,_ = inspect.getsourcelines(f)
    uncommented= [l.split('#')[0] for l in source_lines]
    striped = [l.rstrip() for l in uncommented]
    non_empty = [l for l in striped if l!='']
    return hash(''.join(non_empty))

class Parameters(dict):
    def __init__(self,**kwargs):
        for k,v in kwargs.iteritems():
            if type(v)==list:
                v = tuple(v)
            self[k]=v
            setattr(self,k,v)

class ModelsMaster(object):
    def __init__(self,folder):
        """
        id is a human readable identifier for the model of the form name-version1.version2
        hash is a unique hash created from the model
        metadata contains the same info as the hash but is a dictionary (non hashable)
        """
        saves_name = 'saves'
        summaries_name='summaries'
        models_name = 'models.json'
        self._init_folder(folder,saves_name,models_name,summaries_name)
        self.folder = folder
        self.saves_folder = folder+'/'+saves_name
        self.summaries_folder = folder+'/'+summaries_name
        self.models_file = folder+'/'+models_name
        self.models = {}
        self._hash_to_id = {}
        self.models_metadata = {}
        self._run = self._init_run()
        self._init_models()
        
    def _init_folder(self,folder,saves_name,models_name,summaries_name):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        if not os.path.isdir(folder+'/'+saves_name):
            os.mkdir(folder+'/'+saves_name)
        if not os.path.isdir(folder+'/'+summaries_name):
            os.mkdir(folder+'/'+summaries_name)
        if not os.path.isfile(folder+'/'+models_name):
            with open(folder+'/'+models_name,'wb') as f:
                f.write('{}')

    @staticmethod
    def _hash_metadata(metadata):
        model_hash = (
            metadata['name'],
            metadata['architecture_id'],
            frozenset(tuple(metadata['parameters'].items()))
            )
        return model_hash

    @staticmethod
    def _metadata_from_model(model):
        methods_hashes = tuple(hash_function(getattr(model,f)) for f in ['_create','_train','_train_step'])
        metadata = {
            'name':model.name,
            'architecture_id':methods_hashes,
            'parameters':model.params
        }
        return metadata

    @staticmethod
    def _model_id_to_string(model_id):
        name,v1,v2 = model_id
        return "{}-{}.{}".format(name,v1,v2)

    @staticmethod
    def _string_to_model_id(string):
        name,versions = string.split('-')
        v1,v2 = versions.split('.')
        return (name,int(v1),int(v2))
    
    def get_summaries_file(self,prefix,model_id):
        return '{}/{}_{}_{}'.format(
            self.summaries_folder,
            self._model_id_to_string(model_id),
            prefix,
            self._run)
    
    def _init_run(self):
        return max([0]+[int(f_name.split('_')[-1]) for f_name in os.listdir(self.summaries_folder)])
            
    def _init_models(self):
        """Reads models file to initialise class"""
        
        with open(self.models_file,'rb') as f:
            models_dict = json.load(f)
        for string_id,metadata in models_dict.iteritems():
            model_id = self._string_to_model_id(string_id)
            metadata['parameters'] = Parameters(**metadata['parameters'])
            metadata['architecture_id'] = tuple(metadata['architecture_id'])
            self.models[model_id] = None
            self.models_metadata[model_id] = metadata
            self._hash_to_id[self._hash_metadata(metadata)]=model_id
                
    def _update_json(self,model_id,metadata):
        with open(self.models_file,'rb') as f:
            models_dict = json.load(f)
        models_dict[self._model_id_to_string(model_id)] = metadata
        with open(self.models_file,'wb') as f:
            json.dump(models_dict,f,indent=4)

    def get_new_model_id(self,model_hash):
        """The model_id is a tuple (name,v1,v2)"""
        name,architecture_hash,parameters_hash = model_hash
        hash_ids = self._hash_to_id.items()
        if name not in [hi[0][0] for hi in hash_ids]:
            return (name,0,0)
        hash_ids = [hi for hi in hash_ids if hi[0][0] == name]
        if architecture_hash not in [hi[0][1] for hi in hash_ids]:
            return (name,max([hi[1][1] for hi in hash_ids])+1,0)
        hash_ids = [hi for hi in hash_ids if hi[0][1] == architecture_hash]
        architecture_version = hash_ids[0][1][1]
        if parameters_hash not in [hi[0][2] for hi in hash_ids]:
            return (name,architecture_version,max([hi[1][2] for hi in hash_ids])+1)
        hash_ids = [hi for hi in hash_ids if hi[0][2] == parameters_hash]
        parameters_version = hash_ids[0][1][2]
        return (name,architecture_version,parameters_version)

    def register(self,model):
        model_metadata = self._metadata_from_model(model)
        model_hash = self._hash_metadata(model_metadata)
        if model_hash not in self._hash_to_id.keys():
            print '-'*30
            print 'Registering new model'
            print '-'*30
            # model_id = max(self.models.keys())+1 if self.models else 0
            model_id = self.get_new_model_id(model_hash)
            print 'Model id: {}'.format(model_id)
            self.models[model_id] = model
            self.models_metadata[model_id] = model_metadata
            for k,v in model_metadata.iteritems():
                print k.upper().replace('_',' ')
                print v
            self._update_json(model_id,model_metadata)
            self._hash_to_id[model_hash] = model_id
        else:
            print '-'*30
            print 'Model already registered'
            print '-'*30
            model_id = self._hash_to_id[model_hash]
            self.models[model_id] = model
    
        return model_id
    
    def get_run(self):
        return self._run
    
    def new_run(self):
        self._run+=1
    
    def save_graph(self,model_id,force=False):
        name = self.get_save_graph_folder(model_id)
        if os.path.isfile(name) and not force:
            return
        with self.models[model_id].sess.graph.as_default():
            tf.train.export_meta_graph(name)
        
    def save_variables(self,model_id):
        if not os.path.isdir(self.get_save_variables_folder(model_id)):
            os.mkdir(self.get_save_variables_folder(model_id))
        with self.models[model_id].sess.graph.as_default():
            saver = tf.train.Saver()
            saver.save(
                self.models[model_id].sess,
                self.get_save_variables_file(model_id)
            )
        
    def restore(self,model_id,restore_graph):
         
        model = self.models[model_id]
        with model.sess.graph.as_default():
            if restore_graph:
                saver = tf.train.import_meta_graph(self.get_save_graph_folder(model_id))
            else:
                saver = tf.train.Saver()
            if not os.path.isfile(self.get_save_variables_file(model_id)):
                print 'No saved variables for this step'
                return
            saver.restore(model.sess,self.get_save_variables_file(model_id))
            if restore_graph:
                model.graph = model.sess.graph
        
    def get_save_graph_folder(self,model_id):
        return '{}/graph_{}'.format(self.saves_folder,self._model_id_to_string(model_id))

    def get_save_variables_name(self,model_id):
        return 'vars_{}_{}'.format(
            self._model_id_to_string(model_id), 
            str(self.models[model_id]._step)
            )
    
    def get_save_variables_folder(self,model_id):
        name = self.get_save_variables_name(model_id)
        return '{}/{}'.format(
            self.saves_folder, 
            name
            )
    
    def get_save_variables_file(self,model_id):
        name = self.get_save_variables_name(model_id)
        return '{}/{}'.format(
            self.get_save_variables_folder(model_id), 
            name
            )
    
    def get_max_step(self,model_id):
        return max(
            [0]+[
                int(fname.split('_')[2])
                for fname 
                in os.listdir(self.saves_folder) 
                if fname.split('_')[0]=='vars'
                and fname.split('_')[1]==self._model_id_to_string(model_id)
            ])

class Model(object):
    def __init__(self,name=None,testing=False,**kwargs):
        self.name = self.__class__.__name__ if name is None else name
        self.params = Parameters(**kwargs)
        self._master = _globals['master']
        self._step = 0
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        if not testing:
            self._id = self._master.register(self)
        with self.sess.graph.as_default():
            self._create(self.params)
        if not testing:
            self._save_graph(force=False)
        
    @property
    def train_summaries_dir(self):
        return self._master.get_summaries_file('train',self._id)#'train_%s'%self._master.get_run()
    
    @property
    def test_summaries_dir(self):
        return self._master.get_summaries_file('test',self._id)
    
    def _create(self,params):
        """Saves everything needed"""
        raise NotImplemented
    
    def _train(self):
        raise NotImplemented
        
    def train(self,data,iterations,restart=False):
        if restart:
            self._step = 0
        if self._step == 0:
            self._master.new_run()
        with self.sess.graph.as_default():
            for i in range(iterations):
                self._train_step(data,self._step,self.params)
                self._step+=1
            
        self.save()
        
    def restore(self,step=None,restore_graph=False):
        if step is None:
            step = self._master.get_max_step(self._id)
        self._step = step
        self._master.restore(self._id,restore_graph)
        
    def save(self):
        self._master.save_variables(self._id)
        
    def _save_graph(self,force=False):
        self._master.save_graph(self._id,force=force)
        
    def _train_step(self,data,step,params):
        raise NotImplemented

class ExampleModel(Model):
    def _create(self,params):
        
        self.x = tf.placeholder(tf.float32,shape=(None,2),name='x')
        
        W = tf.Variable(tf.truncated_normal(shape=(2,1)),name='W')
        
        y = tf.matmul(self.x,W,name='y')
        
        self.y_ = tf.placeholder(tf.float32,shape=(None,1),name='y_')
        
        self.cost = tf.nn.l2_loss(y-self.y_)
        
        self.cost_summary = tf.scalar_summary('cost',self.cost)
        
        self.train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(self.cost)
        
        init=tf.initialize_all_variables()
        
        self.train_writer = tf.train.SummaryWriter(self.train_summaries_dir,self.sess.graph)
        self.test_writer = tf.train.SummaryWriter(self.test_summaries_dir)
        
        self.sess.run(init)
    
        
    def _train_step(self,data,step,params):
        
        x_train_split, y_train_split, x_test_split, y_test_split = data
        
        perm = np.random.choice(range(len(x_train_split)),params.batch_size,replace=False)
        batch_x = x_train_split[perm]
        batch_y = y_train_split[perm]
        if step%100 == 0:
            perm_test = np.random.choice(range(len(x_test_split)),params.batch_size,replace=False)
            batch_x_test = x_test_split[perm_test]
            batch_y_test = y_test_split[perm_test]

            summary_test,cost_test = self.sess.run([self.cost_summary,self.cost],feed_dict={self.x: batch_x_test,self.y_: batch_y_test.reshape((-1,1))})
            summary_train,cost_train = self.sess.run([self.cost_summary,self.cost],feed_dict={self.x: batch_x, self.y_: batch_y.reshape((-1,1))})
            print("step %d, train cost %.3f, test cost %.3f"%(step, cost_train, cost_test))
            
        _ = self.sess.run(self.train_step,feed_dict={self.x: batch_x, self.y_: batch_y.reshape((-1,1))})


def init(folder):
    _globals['master'] = ModelsMaster(folder)

