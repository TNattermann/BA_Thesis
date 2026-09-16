import boto3
import json
import translators as ts
import pandas as pd
from ada_inference import infere_ada_002
import glob
from tqdm import tqdm
from io import BytesIO
import time

s3 = boto3.client('s3')
BUCKET_NAME = 'raw-data'


def load_from_s3(object_key):
    response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
    pickle_content = response['Body'].read()
    with BytesIO(pickle_content) as bio:
        df = pd.read_pickle(bio)
    return df

def translate(text, from_language='en', to_language='de', translator='bing'):
    translation = ts.translate_text(query_text=text, translator=translator, from_language=from_language,
                                    to_language=to_language)
    return translation


def concat_batches():
    file_paths = sorted(glob.glob(f'data/*.pkl'))
    dfs = [pd.read_pickle(file) for file in file_paths]
    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df.to_pickle(f'data/*.pkl')


def max_vote(row, threshold):
    max_value = row.max()
    if max_value >= threshold:
        return row.idxmax()
    else:
        return 'no emotion'


class DataLoader:
    """
    Class used to preform ada embedding on dataset
    """

    def load_data(self, data):
        object_key = f'{data}/data.jsonl'
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
        jsonl_content = response['Body'].read().decode('utf-8')
        data = []
        keys = ['id', 'source', 'text']
        for line in jsonl_content.splitlines():
            entry = json.loads(line)
            cleaned = {key: entry[key] for key in keys}
            cleaned['emotion'] = entry['gold']['emotion']
            data.append(cleaned)
        return data

    def formathandler(self):
        batch_size = 50
        raw = self.load_data()
        df = pd.DataFrame(raw)
        resampled_df = df.groupby('emotion').apply(lambda x: x.sample(n=min(230, len(x)), random_state=42)).set_index(
            'id').sort_values(by='id') # remove some of the neutral entries
        for batch_idx in range(0, len(resampled_df), batch_size):
            batch = resampled_df.iloc[batch_idx:batch_idx + batch_size].copy()
            batch['ada_embedding'] = infere_ada_002(batch.text.tolist())
            file_path = f'{batch_idx}.pkl'
            batch.to_pickle(file_path)
            print(f'Success in processing batch from {batch_idx} to {batch_idx + batch_size}')
        concat_batches()


    def translate(self):
        """
        :return: writes translation to csv
        """
        df = pd.DataFrame(self.load_data())
        df = df.groupby('emotion').apply(lambda x: x.sample(n=min(230, len(x)), random_state=42)).set_index(
            'id').sort_values(by='id')
        batch_size = 50
        # Iterate through the DataFrame rows and perform translations
        skipped = []
        for batch_idx in range(0, len(df), batch_size):
            result_data = []
            batch = df.iloc[batch_idx:batch_idx + batch_size].copy()
            for index, row in tqdm(batch[['text']].iterrows(), total=len(batch)):
                id_value = index
                original = row['text']
                # Perform translations
                try:
                    bing = translate(original, from_language='de', to_language='en', translator='bing')
                    result_data.append(
                        {'id': id_value, 'text': original, 'bing': bing})
                except:
                    print('1st error')
                    time.sleep(1.5)
                    try:
                        bing = translate(original, from_language='de', to_language='en', translator='bing')
                        result_data.append(
                            {'id': id_value, 'text': original, 'bing': bing})
                    except:
                        print('2nd error, skipping case')
                        skipped.append(id_value)
            print(f'translation complete for batch{batch_idx}, store to csv\n------')
            # Create a DataFrame from the list of dictionaries
            result_df = pd.DataFrame(result_data)

            # Write the result DataFrame to a CSV file
            result_df.to_csv(f'../data/Translation_retry.csv', index=False)
        with open('../data/skipped.json', 'w') as fp:
            json.dump(skipped, fp)

    def retranslation(self):
        """
        :return: writes retranslation to csv
        """
        df = pd.read_csv('../data/Translation.csv')
        with open('../data/skipped.json', 'r') as json_file:
            skipped = json.load(json_file)
        df = df.loc[skipped]
        batch_size = 50
        # Iterate through the DataFrame rows and perform translations
        skipped = []
        for batch_idx in range(0, len(df), batch_size):
            result_data = []
            batch = df.iloc[batch_idx:batch_idx + batch_size].copy()
            for index, row in tqdm(batch[['bing']].iterrows(), total=len(batch)):
                id_value = index
                original = row['bing']
                # Perform translations
                try:
                    bing = translate(original, from_language='en', to_language='de', translator='bing')
                    result_data.append(
                        {'id': id_value, 'retranslation': bing})
                except:
                    print('1st error')
                    time.sleep(1.5)
                    try:
                        bing = translate(original, from_language='en', to_language='de', translator='bing')
                        result_data.append(
                            {'id': id_value, 'retranslation': bing})
                    except:
                        print('2nd error, skipping case')
                        skipped.append(id_value)
            print(f'translation complete for batch{batch_idx}, store to csv\n------')
            # Create a DataFrame from the list of dictionaries
            result_df = pd.DataFrame(result_data)

            # Write the result DataFrame to a CSV file
            result_df.to_csv(f'../data/Retranslation_retry1_{batch_idx}.csv', index=False)
        with open('../data/skipped.json', 'w') as fp:
            json.dump(skipped, fp)
