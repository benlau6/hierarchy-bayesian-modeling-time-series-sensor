import pandas as pd
import boto3
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Data():
    
    def __init__(self, bucket, prefix, last_n_days=30):
        self.last_n_days = last_n_days
        self.bucket = bucket
        self.prefix = prefix
        self.unique_attrs = ['linename', 'subtrackname', 'Aligned Chainage', 'vehicle']
        self.useful_cols = ['linename', 'subtrackname', 'km', 'vehicle', 'datetime',
                            'acc', 'lat_jerk', 'lon_jerk', 'gauge', 'bolaccy', 'Aligned Chainage']
        self.df_raw = None
        self.df_processed = None
        self.df_result = None
        self.dates = []
        self.files = []
        self._get_path_list()
        
    def _get_path_list(self):
        s3 = boto3.resource('s3', region_name='ap-east-1')
        objs = list(s3.Bucket(self.bucket).objects.filter(Prefix=self.prefix))
        for i in range(1, len(objs)):
            self.files.append(objs[i].key)
            self.dates.append(objs[i].key.rsplit('/',4)[3].rsplit('.',2)[0])
        
    def _get_raw_generator(self):
        for date in self.dates[-self.last_n_days:]:
            df = pd.read_parquet(f's3://{self.bucket}/{self.prefix}{date}.parquet', columns=self.useful_cols)
            yield df
            
    def _get_max_sub_df(self, df, attr):
        attr_max = df[attr].value_counts().index[0]
        mask = df[attr] == attr_max
        return df[mask]
    
    @classmethod
    def get_one_sample(cls, bucket, prefix):       
        data = cls(bucket, prefix, 1)
        df = pd.read_parquet(f's3://{data.bucket}/{data.prefix}{data.dates[-1]}.parquet')
        return df

    @staticmethod
    def check_na(df, threshold=0.3):       
        df_na = df.isna().sum()/df.shape[0]
        return df_na[df_na>=threshold]
            
    def get_raw_generator(self):
        self.df_raw = pd.concat(list(self._get_raw_generator()))
        self.df_processed = self.df_raw
    
    def get_example(self):
        for attr in self.unique_attrs:
            self.df_processed = self._get_max_sub_df(self.df_processed, attr)
        
    def preprocessing(self):
        self.df_processed = self.df_processed.drop_duplicates('datetime').sort_values(by='datetime')
        #self.df_processed['gauge'] = self.df_processed['gauge'].interpolate(limit=2)
        
    def scatter_plot_chainage_hue(self, chainage=2.1, direction='Down'):            
        mask1 = self.df_processed['Aligned Chainage'] == chainage
        mask2 = self.df_processed['subtrackname'].str.contains(direction)
        df = self.df_processed[mask1&mask2]
        fig, axes = plt.subplots(2,3, figsize=(16, 8))
        fig.suptitle(f'chainage={chainage}, direction={direction}')
        sns.scatterplot(data=df, hue='km', x='datetime', y='acc', ax=axes[0, 0])#.set_title('acc')
        sns.scatterplot(data=df, hue='km', x='datetime', y='bolaccy', ax=axes[0, 1])#.set_title('bolaccy')
        sns.scatterplot(data=df, hue='km', x='datetime', y='gauge', ax=axes[0, 2])#.set_title('gauge')
        sns.scatterplot(data=df, hue='vehicle', x='datetime', y='acc', ax=axes[1, 0])#.set_title('acc')
        sns.scatterplot(data=df, hue='vehicle', x='datetime', y='bolaccy', ax=axes[1, 1])#.set_title('bolaccy')
        sns.scatterplot(data=df, hue='vehicle', x='datetime', y='gauge', ax=axes[1, 2])#.set_title('gauge')
        fig.autofmt_xdate()
        plt.show()
            
    def sub_df(self, chainage=2.1, direction='Down', target='gauge'):
        mask1 = self.df_processed['Aligned Chainage'] == chainage
        mask2 = self.df_processed['subtrackname'].str.contains(direction)
        df = self.df_processed[mask1&mask2]
        df = df[['datetime', target]]
        df = df.set_index('datetime')
        return df
    