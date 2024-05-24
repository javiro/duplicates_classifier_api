import pickle
import pandas as pd
import sqlite3

from fuzzywuzzy import fuzz
from ray import serve
from starlette.requests import Request


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class DuplicatesClassifier:

    features = [
        'partial_ratio',
        'partial_token_set_ratio',
        'partial_token_sort_ratio',
        'ratio',
        'token_set_ratio',
        'token_sort_ratio'
    ]

    fields = ['artists', 'contributors', 'title']
    
    def __init__(self):
        # Load model
        filename = 'best_model.sav'
        self.model = pickle.load(open(filename, 'rb'))
        self.conn = sqlite3.connect("db.db")
    
    def add_similarity_feature(self, df_data):
        new_features = []
        for function in self.features:
            for field in self.fields:
                assert field + '_q' in df_data.columns
                fuzz_function = getattr(fuzz, function)
                df_data[f'{field}_{function}'] = df_data.apply(lambda x: fuzz_function(x[f'{field}_q'], x[f'{field}_m']), axis=1)
                new_features.append(f'{field}_{function}')
        return df_data, new_features
    
    def join_sr(self, groundtruth, soundrecording):
        data = groundtruth.merge(soundrecording, right_on='sr_id', left_on='q_source_id', how='inner').merge(soundrecording, right_on='sr_id', left_on='m_source_id', how='inner')
        data.columns = [c.replace('_x', '_q').replace('_y', '_m') for c in data.columns]
        return data

    def get_features(self, input_string):
        q_sr_id, m_sr_id = [item.split('=')[1] for item in input_string.split('&')]
        soundrecording = pd.read_sql_query(f"SELECT * FROM soundrecording where sr_id in ('{q_sr_id}', '{m_sr_id}')", self.conn)
        groundtruth = pd.DataFrame({'q_source_id': [q_sr_id], 'm_source_id': [m_sr_id]})
        data = self.join_sr(groundtruth, soundrecording)
        new_data, new_features = self.add_similarity_feature(data)
        new_data['isrcs_coincidence'] = (new_data['isrcs_m'] == new_data['isrcs_q']).astype(int)
        return new_data[[*new_features, *['isrcs_coincidence']]]

    @staticmethod
    def check_input_string(input_string):
        return all([item in input_string for item in ['q_sr_id=', '&', 'm_sr_id=']])
        
    def get_api_response(self, input_string):
        if not self.check_input_string(input_string):
            return {"error": "Incorrect request format. Please use q_sr_id= & m_sr_id="}
        else:
            X = self.get_features(input_string)
            return {"class": "valid" if self.model.predict(X) == 1 else "invalid"}

    async def __call__(self, http_request: Request) -> dict:
        input_string: str = await http_request.json()
        return self.get_api_response(input_string)


duplicate_classifier_app = DuplicatesClassifier.bind()
