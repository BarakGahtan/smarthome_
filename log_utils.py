import os


class Logger:

    def __init__(self, file_name, headers, overwrite):
        self.log_file_name = file_name
        self.headers = headers
        if overwrite and os.path.isfile(file_name):
            os.remove(file_name)
        if (os.path.isfile(file_name) and os.path.getsize(file_name) > 0) or headers is None:
            return
        msg = ''
        for header in self.headers:
            msg += f'{header},'

        with open(self.log_file_name, 'a') as f:
            f.write(f'{msg[:-1]}\n')

    def log(self, msg_dict):
        with open(self.log_file_name, 'a') as f:
            msg = ''
            for header in self.headers:
                if header not in msg_dict:
                    print(f'WARNING {header} was not logged')
                    continue

                cell_content = msg_dict[header]
                if type(cell_content) is float:
                    msg += f'{cell_content: .5f},'
                else:
                    msg += f'{cell_content},'

            print(msg)
            f.write(f'{msg[:-1]}\n')

    def log_msg(self, msg):
        with open(self.log_file_name, 'a') as f:
            f.write(f'{msg}\n')

    def get_df(self):
        import pandas as pd
        return pd.read_csv(self.log_file_name)


# utility for timing execution
class Timer:
    import time
    timer = time.time

    def __init__(self, msg='time elapsed'):
        self._timer = self.timer
        self._start = 0
        self._end = 0
        self.msg = msg

    def __enter__(self):
        self._start = self._timer()
        return self._timer

    def __exit__(self, a, b, c):
        self._end = self._timer()
        print(f'{self.msg} is : {self._end - self._start}')