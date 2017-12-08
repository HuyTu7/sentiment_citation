import sys
import csv
import pandas as pd
import pickle

class ReadData:
    def __init__(self):
        self.header = []
        self.data = {}

    def format(self, val):
        try:
            fval = float(val)
            if fval.is_integer():
                return int(fval), False
            else:
                return fval, False
        except ValueError:
            return val, True

    def remove_mis(self, line):
        line = line.strip('\n')
        line = "".join(line.split())
        return line

    def read_header(self, h_line):
        entries = self.remove_mis(h_line).split(",")
        for entry in entries:
            entry = entry.strip()
            if entry:
                self.header.append(entry)
                self.data[entry] = []

    def filter(self, row):
        str = []
        text = ""
        quoting_char = False
        p_char = ""
        len_of_string = len(row)
        for i in range(0, len_of_string):
            c_char = row[i]
            if c_char == "'" and not quoting_char:
                quoting_char = True
            elif p_char == "'" and c_char == "," and len(text) > 1 and quoting_char:
                quoting_char = False

            if (c_char != "," or quoting_char) and i != len_of_string - 1:
                text += c_char
            else:
                if text.startswith("'") and text.endswith("'"):
                    text = text[1:-1]
                elif i == len_of_string - 1:
                    print text
                    text = text[1:]
                    print text
                str.append(text)
                text = ""
            p_char = c_char

        return str


    def read_arff(self, filename):
        with open(filename, "rb") as f2r:
            self.read_header(f2r.readline())
            data = []
            row = f2r.readline()
            count = 0
            while row:
                count += 1
                data.append(self.filter(row))
                row = f2r.readline()
        print count
        result = self.weird(data)
        self.csv_operating("test_data", result)


    def max_min(self, array):
        nmax = -sys.maxint
        nmin = +sys.maxint
        sum = 0
        count = 0
        for i in array:
            number = len(i)
            if number > nmax:
                nmax = number
            if number < nmin:
                nmin = number
            if number != 5:
                print i[1]
                count += 1
            sum += number
        avg = sum/len(array)
        print count
        print avg, nmax, nmin

    def weird(self, array):
        result = []
        for i in array:
            if len(i) == 5:
                result.append(i)
        print len(result)
        return result

    def csv_operating(self, fname, list_item):
        # save or load the pickle file.
        file_name = '%s.csv' % fname
        print(file_name)
        if not list_item:
            df = pd.read_csv(file_name, sep='\t')
            return df
        else:
            df = pd.DataFrame(list_item, columns=["class", "id", "text", "author", "dependencies"])
            df.to_csv(file_name, sep='\t')

    def pickle_operating(self, fname, item, flag):
        # save or load the pickle file.
        file_name = '%s.pickle' % fname
        print(file_name)
        if flag == 1:
            with open(file_name, 'rb') as fs:
                item = pickle.load(fs)
                return item
        else:
            with open(file_name, 'wb') as fs:
                pickle.dump(item, fs, protocol=pickle.HIGHEST_PROTOCOL)




