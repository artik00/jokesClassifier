import os
import git


class DataLoader:

    def __init__(self, url, directory_to_clone_to, file_to_extract = None):
        self.url_to_clone = url
        self.dir = directory_to_clone_to
        self.copy_data_set()



    def copy_data_set(self):
        if not os.path.exists(self.dir):
            print("fetching the data source from url...")
            git.Repo.clone_from(self.url_to_clone, self.dir, branch='master')
        else:
            print('data already exists')


