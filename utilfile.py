import codecs


class StringUtil:

    @staticmethod
    def remove_ne(message):
        return message.replace('<NE>', '').replace('</NE>', '').replace('<AB>', '').replace('</AB>', '')
    

class FileUtil:

    def n_write_file(self, file_name, message):
        print message
        f = open(file_name,'a+b')
        f.write(message+'\n')
        f.close()
        
    def write_file(self, file_name, message):
        with codecs.open(file_name, 'a+b', encoding='utf8') as f:
            f.write(message + '\n')
            f.close()

    def write_file2(self, name, message):
        f = open(name,'w')
        f.write(message)
        f.close()
        
    def write_input_file2(self, data, path):
        str = '1'
        str += ' 1:{}'.format(data['share'])
        str += ' 2:{}'.format(data['like_number'])
        str += ' 3:{}'.format(data['url'])
        str += ' 4:{}'.format(data['vdo'])
        str += ' 5:{}'.format(data['image'])
        str += ' 6:{}'.format(data['comment'])
        str += ' 7:{}'.format(data['is_location'])
        str += ' 8:{}'.format(data['hashtag'])
        print 'str utl ', str
        self.write_file('/tmp/test', str)
        
    def write_file_no_n(self, file_name, message):
         with codecs.open(file_name, 'a+b', encoding='utf8') as f:
            f.write(message)
            f.close()
    
    def write_newfile_n(self, file_name, message):
           self.template_write_file(file_name, message, 'w')
         
    def write_newfile(self, file_name, message):
           self.template_write_file(file_name, message, 'w')
    
    def read_noencode(self, filename):
        f = open(filename)
        lines = f.readlines()
        f.close()
        return lines

    def read_result(self, path):
        str = self.read_file_one_line(path)
        return str
    
    def read_file_one_line(self, path):
        f = open(path, 'r')
        msg = f.read()
        f.close()
        return msg
         
    def read_file(self, file_name):
        with codecs.open(file_name, 'r', 'utf-8') as file:
            return file.readlines()
        return []
    
    def template_write_file(self, file_name, message, option):
          with codecs.open(file_name, option, encoding='utf8') as f:
            f.write(message)
            f.close()       
