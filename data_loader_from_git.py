import os
import git


class DataLoader:
    """
    This class loads data from git , git clone :
    """
    def __init__(self, url, directory_to_clone_to):
        """
        :param url: git url
        :param directory_to_clone_to: which dir in the git repo to upload
        """
        self.url_to_clone = url
        self.dir = directory_to_clone_to
        self.copy_data_set()

    def copy_data_set(self):
        if not os.path.exists(self.dir):
            print("fetching the data source from url...")
            git.Repo.clone_from(self.url_to_clone, self.dir, branch='master')
        else:
            print('data already exists')
