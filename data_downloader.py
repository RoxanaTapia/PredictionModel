import os
import requests

"""
Retrieves complete c signals from PhysioNet data bank
NOTE THAT you need to manually correct some .hea files in 103 and 222 (some non-utf8 characters are in the text)
"""


def write(filename, content):
    try:
        open(filename, 'r')
    except FileNotFoundError:
        print('Writing data: {filename}'.format(filename=filename))
        file = open(filename, 'wb')
        with file as f:
            f.write(content)
            f.close()
        return
    print('File already exist: {filename}'.format(filename=filename))


def get_data(set_name, event, number_of_tests):
    base_url = 'https://physionet.org/physiobank/database/challenge/2009/test-set-{set_name}/{event}{test_number}c/'
    specific_urls = [
        '{event}{test_number}c.hea',
        '{event}{test_number}c_layout.hea',
        '{event}{test_number}cn.dat',
        '{event}{test_number}cn.hea']
    global_urls = [
        '{event}{test_number}c_{identifier}.dat',
        '{event}{test_number}c_{identifier}.hea'
    ]

    # for i in range(1, number_of_tests+1):
    for i in range(3, 4):
        test_number = str(i).zfill(2)
        base = base_url.format(event=event, set_name=set_name, test_number=test_number)

        j = 1
        while True:
            identifier = str(j).zfill(4)
            global_url1 = global_urls[0].format(event=event, test_number=test_number, identifier=identifier)
            global_url2 = global_urls[1].format(event=event, test_number=test_number, identifier=identifier)
            url1 = base + global_url1
            url2 = base + global_url2

            print('Requesting resource: {resource}'.format(resource=url1))
            print('Requesting resource: {resource}'.format(resource=url2))
            response1 = requests.get(url1)
            response2 = requests.get(url2)
            j = j + 1
            if response1.status_code == 200 and response2.status_code == 200:
                directory = 'resources/c_signals/test-set-{set_name}/{event}{test_number}c/'.format(
                    event=event,
                    set_name=set_name,
                    test_number=test_number
                )
                if not os.path.exists(directory):
                    os.makedirs(directory)
                path1 = directory + global_url1
                path2 = directory + global_url2
                write(path1, response1.content)
                write(path2, response2.content)

            elif response1.status_code == 404 and response2.status_code == 404:
                print("Finished.")
                break
            else:
                print("Something went wrong:\n")
                if response1.status_code != 200:
                    print("{status_code}. {resource}".format(status_code=response1.status_code, resource=url1))
                if response2.status_code != 200:
                    print("{status_code}. {resource}".format(status_code=response1.status_code, resource=url1))
                break

        for url in specific_urls:
            uri = base + url.format(event=event, test_number=test_number)
            print('Requesting resource: {resource}'.format(resource=uri))
            response = requests.get(uri)

            if response.status_code == 200:
                path = 'resources/c_signals/test-set-{set_name}/{event}{test_number}c/'.format(
                    event=event,
                    set_name=set_name,
                    test_number=test_number) + url.format(event=event, test_number=test_number)
                write(path, response.content)
            else:
                print("Something went wrong:\n")
                print("{status_code}. {resource}".format(status_code=response.status_code, resource=uri))
        print("Finished.\n")

if __name__ == '__main__':

    get_data(set_name='a', event=1, number_of_tests=10)
    get_data(set_name='b', event=2,  number_of_tests=40)




