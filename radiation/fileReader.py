import fnmatch
import os
import random
import re
import threading
import json

import tensorflow as tf
from netCDF4 import Dataset
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.interpolate import splev, splrep

"""
Data v7: statistical values
ST:  min 100.0000000000 max 333.1499946801 mean 268.1406929063 std 37.5368706325
C:  min 0.0000000000 max 0.0099999904 mean 0.0017284657 std 0.0023850203
R:  min -255.9600440474 max 78.3198382662 mean -1.3758656541 std 6.1112494507
T:  min 100.0000000000 max 355.5721906214 mean 230.1788102309 Td 46.5063403685
H:  min -2720.3344538111 max 1848.3667831706 mean 4.2050377031 Hd 13.4852605066
"""

#
#minT = 178.87485
minT = 150
#maxT = 310.52261
maxT = 340

minC = 0
maxC = 0.01

#minH = 6.324828e-08
minH = 0
#maxH = 0.021951281
maxH = 0.1

minR = -59.08844
maxR = 14.877947

minP = 0.0
maxP = 103000


epoch = 0

lock = threading.Lock()

standard_x = [0.0, 1111.111111111111, 2222.222222222222, 3333.333333333333, 4444.444444444444, 5555.555555555556, 6666.666666666666, 7777.777777777777, 8888.888888888889, 10000.0, 11000.0, 13875.0, 16750.0, 19625.0, 22500.0, 25375.0, 28250.0, 31125.0, 34000.0, 36875.0, 39750.0, 42625.0, 45500.0, 48375.0, 51250.0, 54125.0, 57000.0, 59875.0, 62750.0, 65625.0, 68500.0, 71375.0, 74250.0, 77125.0, 80000.0, 82760.0, 83603.33333333333, 84446.66666666667, 85290.0, 86133.33333333333, 86976.66666666667, 87820.0, 88663.33333333333, 89506.66666666667, 90350.0, 91193.33333333333, 92036.66666666667, 92880.0, 93723.33333333333, 94566.66666666667, 95410.0, 96253.33333333333, 97096.66666666667, 97940.0, 98783.33333333333, 99626.66666666667, 100470.0, 101313.33333333334, 102156.66666666667, 103000.0]

standard_x = [2.0, 208.40881763527054, 414.8176352705411, 621.2264529058116, 827.6352705410821, 1034.0440881763527, 1240.4529058116232, 1446.8617234468938, 1653.2705410821643, 1859.6793587174348, 2066.0881763527054, 2272.496993987976, 2478.9058116232463, 2685.314629258517, 2891.7234468937877, 3098.132264529058, 3304.5410821643286, 3510.9498997995993, 3717.3587174348695, 3923.76753507014, 4130.176352705411, 4336.5851703406815, 4542.993987975952, 4749.402805611222, 4955.811623246493, 5162.220440881763, 5368.629258517034, 5575.038076152305, 5781.446893787575, 5987.855711422845, 6194.264529058116, 6400.6733466933865, 6607.082164328657, 6813.490981963928, 7019.899799599199, 7226.308617234469, 7432.717434869739, 7639.12625250501, 7845.53507014028, 8051.943887775551, 8258.352705410822, 8464.761523046092, 8671.170340681363, 8877.579158316634, 9083.987975951904, 9290.396793587173, 9496.805611222444, 9703.214428857715, 9909.623246492985, 10116.032064128256, 10322.440881763527, 10528.849699398797, 10735.258517034068, 10941.667334669339, 11148.07615230461, 11354.48496993988, 11560.89378757515, 11767.302605210421, 11973.71142284569, 12180.120240480961, 12386.529058116232, 12592.937875751502, 12799.346693386773, 13005.755511022044, 13212.164328657314, 13418.573146292585, 13624.981963927856, 13831.390781563126, 14037.799599198397, 14244.208416833668, 14450.617234468938, 14657.026052104207, 14863.434869739478, 15069.843687374749, 15276.25250501002, 15482.66132264529, 15689.07014028056, 15895.478957915831, 16101.887775551102, 16308.296593186373, 16514.705410821643, 16721.114228456914, 16927.523046092185, 17133.931863727455, 17340.340681362726, 17546.749498997997, 17753.158316633268, 17959.56713426854, 18165.97595190381, 18372.38476953908, 18578.793587174347, 18785.202404809617, 18991.611222444888, 19198.02004008016, 19404.42885771543, 19610.8376753507, 19817.24649298597, 20023.65531062124, 20230.064128256512, 20436.472945891783, 20642.881763527053, 20849.290581162324, 21055.699398797595, 21262.108216432865, 21468.517034068136, 21674.925851703407, 21881.334669338677, 22087.743486973948, 22294.15230460922, 22500.56112224449, 22706.96993987976, 22913.37875751503, 23119.7875751503, 23326.196392785572, 23532.605210420843, 23739.014028056114, 23945.42284569138, 24151.83166332665, 24358.240480961922, 24564.649298597193, 24771.058116232463, 24977.466933867734, 25183.875751503005, 25390.284569138275, 25596.693386773546, 25803.102204408817, 26009.511022044087, 26215.919839679358, 26422.32865731463, 26628.7374749499, 26835.14629258517, 27041.55511022044, 27247.96392785571, 27454.372745490982, 27660.781563126253, 27867.190380761524, 28073.599198396794, 28280.008016032065, 28486.416833667336, 28692.825651302606, 28899.234468937877, 29105.643286573144, 29312.052104208415, 29518.460921843685, 29724.869739478956, 29931.278557114227, 30137.687374749497, 30344.096192384768, 30550.50501002004, 30756.91382765531, 30963.32264529058, 31169.73146292585, 31376.14028056112, 31582.549098196392, 31788.957915831663, 31995.366733466934, 32201.775551102204, 32408.184368737475, 32614.593186372746, 32821.00200400801, 33027.41082164329, 33233.819639278554, 33440.22845691383, 33646.637274549095, 33853.04609218437, 34059.45490981964, 34265.86372745491, 34472.27254509018, 34678.68136272545, 34885.09018036072, 35091.498997995994, 35297.90781563126, 35504.316633266535, 35710.7254509018, 35917.13426853708, 36123.54308617234, 36329.95190380762, 36536.360721442885, 36742.76953907816, 36949.178356713426, 37155.58717434869, 37361.99599198397, 37568.404809619235, 37774.81362725451, 37981.222444889776, 38187.63126252505, 38394.04008016032, 38600.44889779559, 38806.85771543086, 39013.26653306613, 39219.6753507014, 39426.084168336674, 39632.49298597194, 39838.901803607216, 40045.31062124248, 40251.71943887776, 40458.128256513024, 40664.5370741483, 40870.945891783565, 41077.35470941884, 41283.76352705411, 41490.17234468938, 41696.58116232465, 41902.98997995992, 42109.39879759519, 42315.80761523046, 42522.21643286573, 42728.625250501, 42935.03406813627, 43141.44288577154, 43347.85170340681, 43554.26052104208, 43760.669338677355, 43967.07815631262, 44173.486973947896, 44379.89579158316, 44586.30460921844, 44792.713426853705, 44999.12224448898, 45205.531062124246, 45411.93987975952, 45618.34869739479, 45824.75751503006, 46031.16633266533, 46237.5751503006, 46443.98396793587, 46650.392785571144, 46856.80160320641, 47063.210420841686, 47269.61923847695, 47476.02805611223, 47682.436873747494, 47888.84569138276, 48095.254509018036, 48301.6633266533, 48508.07214428858, 48714.480961923844, 48920.88977955912, 49127.298597194385, 49333.70741482966, 49540.11623246493, 49746.5250501002, 49952.93386773547, 50159.34268537074, 50365.75150300601, 50572.160320641284, 50778.56913827655, 50984.977955911825, 51191.38677354709, 51397.79559118237, 51604.20440881763, 51810.61322645291, 52017.022044088175, 52223.43086172345, 52429.839679358716, 52636.24849699399, 52842.65731462926, 53049.066132264525, 53255.4749498998, 53461.883767535066, 53668.29258517034, 53874.70140280561, 54081.11022044088, 54287.51903807615, 54493.92785571142, 54700.33667334669, 54906.745490981964, 55113.15430861723, 55319.563126252506, 55525.97194388777, 55732.38076152305, 55938.789579158314, 56145.19839679359, 56351.607214428856, 56558.01603206413, 56764.4248496994, 56970.83366733467, 57177.24248496994, 57383.65130260521, 57590.06012024048, 57796.468937875754, 58002.87775551102, 58209.28657314629, 58415.69539078156, 58622.10420841683, 58828.513026052104, 59034.92184368737, 59241.330661322645, 59447.73947895791, 59654.14829659319, 59860.55711422845, 60066.96593186373, 60273.374749498995, 60479.78356713427, 60686.192384769536, 60892.60120240481, 61099.01002004008, 61305.41883767535, 61511.82765531062, 61718.23647294589, 61924.64529058116, 62131.054108216435, 62337.4629258517, 62543.871743486976, 62750.28056112224, 62956.68937875752, 63163.098196392784, 63369.50701402806, 63575.915831663326, 63782.32464929859, 63988.73346693387, 64195.142284569134, 64401.55110220441, 64607.959919839675, 64814.36873747495, 65020.77755511022, 65227.18637274549, 65433.59519038076, 65640.00400801603, 65846.4128256513, 66052.82164328657, 66259.23046092184, 66465.63927855711, 66672.04809619239, 66878.45691382766, 67084.86573146292, 67291.27454909819, 67497.68336673347, 67704.09218436874, 67910.501002004, 68116.90981963927, 68323.31863727455, 68529.72745490982, 68736.13627254509, 68942.54509018036, 69148.95390781562, 69355.3627254509, 69561.77154308617, 69768.18036072144, 69974.5891783567, 70180.99799599199, 70387.40681362725, 70593.81563126252, 70800.22444889779, 71006.63326653307, 71213.04208416834, 71419.4509018036, 71625.85971943887, 71832.26853707415, 72038.67735470942, 72245.08617234469, 72451.49498997995, 72657.90380761524, 72864.3126252505, 73070.72144288577, 73277.13026052104, 73483.53907815632, 73689.94789579159, 73896.35671342685, 74102.76553106212, 74309.17434869739, 74515.58316633267, 74721.99198396794, 74928.4008016032, 75134.80961923847, 75341.21843687375, 75547.62725450902, 75754.03607214428, 75960.44488977955, 76166.85370741483, 76373.2625250501, 76579.67134268537, 76786.08016032063, 76992.48897795592, 77198.89779559118, 77405.30661322645, 77611.71543086172, 77818.124248497, 78024.53306613227, 78230.94188376753, 78437.3507014028, 78643.75951903808, 78850.16833667335, 79056.57715430862, 79262.98597194388, 79469.39478957915, 79675.80360721443, 79882.2124248497, 80088.62124248497, 80295.03006012023, 80501.43887775551, 80707.84769539078, 80914.25651302605, 81120.66533066132, 81327.0741482966, 81533.48296593186, 81739.89178356713, 81946.3006012024, 82152.70941883768, 82359.11823647295, 82565.52705410821, 82771.93587174348, 82978.34468937876, 83184.75350701403, 83391.1623246493, 83597.57114228456, 83803.97995991984, 84010.38877755511, 84216.79759519038, 84423.20641282565, 84629.61523046091, 84836.0240480962, 85042.43286573146, 85248.84168336673, 85455.250501002, 85661.65931863728, 85868.06813627254, 86074.47695390781, 86280.88577154308, 86487.29458917836, 86693.70340681363, 86900.1122244489, 87106.52104208416, 87312.92985971944, 87519.33867735471, 87725.74749498998, 87932.15631262524, 88138.56513026053, 88344.97394789579, 88551.38276553106, 88757.79158316633, 88964.20040080161, 89170.60921843688, 89377.01803607214, 89583.42685370741, 89789.83567134269, 89996.24448897796, 90202.65330661323, 90409.06212424849, 90615.47094188376, 90821.87975951904, 91028.28857715431, 91234.69739478957, 91441.10621242484, 91647.51503006012, 91853.92384769539, 92060.33266533066, 92266.74148296592, 92473.1503006012, 92679.55911823647, 92885.96793587174, 93092.37675350701, 93298.78557114229, 93505.19438877756, 93711.60320641282, 93918.01202404809, 94124.42084168337, 94330.82965931864, 94537.2384769539, 94743.64729458917, 94950.05611222445, 95156.46492985972, 95362.87374749499, 95569.28256513026, 95775.69138276552, 95982.1002004008, 96188.50901803607, 96394.91783567134, 96601.3266533066, 96807.73547094189, 97014.14428857715, 97220.55310621242, 97426.96192384769, 97633.37074148297, 97839.77955911824, 98046.1883767535, 98252.59719438877, 98459.00601202405, 98665.41482965932, 98871.82364729459, 99078.23246492985, 99284.64128256514, 99491.0501002004, 99697.45891783567, 99903.86773547094, 100110.27655310622, 100316.68537074148, 100523.09418837675, 100729.50300601202, 100935.91182364729, 101142.32064128257, 101348.72945891783, 101555.1382765531, 101761.54709418837, 101967.95591182365, 102174.36472945892, 102380.77354709418, 102587.18236472945, 102793.59118236473, 103000.0]

standard_x = np.linspace(1, 200, 100)
standard_x = np.append(standard_x, np.linspace(202, 50000, 100))
standard_x = np.append(standard_x, np.linspace(50500, 105000, 300))
standard_x = standard_x.tolist()

level_size = len(standard_x)

def cal_air_pressure(air_pressure_interface):
    air_pressure = np.empty(60)
    for level in range(len(air_pressure_interface) - 1):
        air_pressure[level] = (air_pressure_interface[level] + air_pressure_interface[level+1])*0.5
    return air_pressure

def normalizeT(t):
    return normalize(t, minT, maxT)


def normalizeH(h):
    return normalize(h, minH, maxH)


def normalizeC(c):
    return normalize(c, minC, maxC)


def normalizeR(r):
    return normalize(r, minR, maxR)


def normalizeP(p):
    return normalize(p, minP, maxP)


def normalize(x, min, max, mean=1, std=1):
    # TODO: add option to choose between min-max of zero-mean normalization
    return (x - min) / (max - min)  # min max normalization
    # return (x - mean) / std  # standardization - zero-mean normalization
    # return x+100


def denormalize(x, mean, std):
    return x * std + mean


def get_category_cardinality(files):
    """ Deprecated: function used before for identifying the samples based in its name
    and calculate the minimum and maximum sample id
    :param files: array of root paths.
    :return: min_id, max_id: int
    """
    file_pattern = r'([0-9]+)\.csv'
    id_reg_expression = re.compile(file_pattern)
    min_id = None
    max_id = None
    for filename in files:
        id = int(id_reg_expression.findall(filename)[0])
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    """ Function that randomizes a list of filePaths.

    :param files: list of path files
    :return: iterable of random files
    """
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.csv'):
    """ Recursively finds all files matching the pattern.

    :param directory:  directory path
    :param pattern: reggex
    :return: list of files
    """

    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def interpolate(x, y, standard_x, t_or_h='r'):
    x = np.flip(x,0)
    y = np.flip(y,0)


    level_size = len(standard_x)
    spl = splrep(x, y)
    for start_index, start in enumerate(standard_x):
        if x[0] <= start:
            start_index = start_index - 1
            break

    for end_index, end in enumerate(standard_x):
        if x[-1] <= end:
            end_index = end_index + 1
            break
    #valid intervals [start_index, end_index)


    interpolated = splev(standard_x[start_index:end_index], spl)
    interpolated[0] = interpolated[1]
    interpolated[-1] = interpolated[-2]

    if t_or_h == 't':
        standard_y = np.append(np.zeros(start_index), normalizeT(interpolated))
        standard_y = np.append(standard_y, np.zeros(level_size - end_index))
    elif t_or_h == 'h':
        standard_y = np.append(np.zeros(start_index), normalizeH(interpolated))
        standard_y = np.append(standard_y, np.zeros(level_size - end_index))
    else:
        standard_y = np.append(np.zeros(start_index), interpolated)
        standard_y = np.append(standard_y, np.zeros(level_size - end_index))

    assert len(standard_y) == level_size

    return standard_y


def load_data_samples(files):
    """ Generator that yields samples from the directory.

    In the latest versions, the files are files where each line is a sample
    in json format. This function basically read each sample of the file and
    normalizes it and generates the data for the model in the desired format.

    :param files: list of files
    :return: iterable that contains the data, the label and the identifier of the sample.
    """

    # previous json loader
    '''
    for filename in files:
        with open(filename) as f:
            id = 0
            for line in f:
                id += 1
                try:
                    input = json.loads(line)
                    # todo normalize the input
                    data = []
                    label = []

                    data.append((input['co2']))
                    data.append((input['surface_temperature']))
                    for i in range(0, len(input['radiation'])):
                        data.append((input['air_temperature'][i]))
                        data.append((input['humidity'][i]))
                        label.append((input['radiation'][i]))

                    # fill last 2 values with 0
                    for _ in range(0, 196 - 194):
                        data.append(0.0)

                    yield data, label, [id]
                except ValueError:
                    print('Value error in file {} and line {}'.format(filename, id))
    '''

    for filename in files:

        with lock:
            f = Dataset(filename, mode='r')
            v = f.variables['radiation_data'][:]
            f.close()
        #ids = np.random.choice(np.arange(v.shape[0]), size = 10, replace=False)
        for id in range(v.shape[0]):
            #data = np.append(v[id,0:122],v[id, 182:243])
            #data = np.append(v[id,0:2],normalizeT(v[id, 2:62]))
            #data = np.append(data, normalizeH(v[id,62:122]))
            #data = np.append(data, normalizeP(v[id,182:243]))
            '''
            data = []

            for i in range(60):
                data.append(normalizeC(v[id, 0]))
                data.append(normalizeT(v[id, 1]))
                data.append(normalizeT(v[id, i+2]))
                data.append(normalizeH(v[id, i + 62]))
                data.append(normalizeP(v[id, i + 182]))
            '''
            #    data = np.append(data, normalizeC(v[id, 0]))
            #    data = np.append(data, normalizeT(v[id, 1]))
            #    data = np.append(data, normalizeT(v[id, i+2]))
            #    data = np.append(data, normalizeH(v[id, i + 62]))
            #    data = np.append(data, normalizeP(v[id, i + 182]))

            air_pressure = cal_air_pressure(v[id, 182:243])
            inter_air_temperature = interpolate(air_pressure, v[id, 2:62], standard_x, 't')
            inter_humidity = interpolate(air_pressure, v[id, 62:122], standard_x, 'h')
            inter_radiation = interpolate(air_pressure, v[id, 122:182], standard_x)

            data = np.append(normalizeC(v[id, 0]), normalizeT(v[id, 1]))
            data = np.append(data, np.zeros(level_size - 2))
            data = np.append(data, inter_air_temperature)
            data = np.append(data, inter_humidity)
            #data = np.append(data, normalizeP(v[id, 182:243]))

            label = np.array(inter_radiation)

            '''
            data.append(v[id,0])
            data.append(v[id,1])


            num_levels = int((v.shape[1]-2)/3)

            for i in range(num_levels):
                data.append(v[id,i+2])
                data.append(v[id,i+98])
                label.append(v[id,i+194])

            for _ in range(0, 196 - 194):
                data.append(np.float32(0.0))

            '''
            if np.isnan(data.sum()) or np.isnan(label.sum()):
                print("NaN found!!!!!")
                continue

            yield data, label, [id]


class FileReader(object):
    """ Background reader that pre-processes radiation files
    and enqueues them into a TensorFlow queue.

    """

    def __init__(self,
                 data_dir,
                 test_dir,
                 coord,
                 n_input=1500,
                 n_output=500,
                 queue_size=5000000,
                 test_percentage=0.2):

        # TODO: Implement a option that enables the usage of a test queue, by default it is
        # enabled here. For implementing this, the flag should be propagated to the several
        # functions that operate with both queues.

        self.data_dir = data_dir
        self.test_dir = test_dir
        self.coord = coord
        self.n_input = n_input
        self.n_output = n_output
        self.threads = []
        self.sample_placeholder_train = tf.placeholder(tf.float32, [n_input])
        self.result_placeholder_train = tf.placeholder(tf.float32, [n_output])
        self.sample_placeholder_test = tf.placeholder(tf.float32, [n_input])
        self.result_placeholder_test = tf.placeholder(tf.float32, [n_output])
        self.idFile_placeholder_test = tf.placeholder(tf.int32, [1])
        self.idFile_placeholder_train = tf.placeholder(tf.int32, [1])

        self.queue_train = tf.PaddingFIFOQueue(queue_size, [tf.float32, tf.float32, tf.int32],
                                               shapes=[[n_input], [n_output], [1]])
        self.queue_test = tf.PaddingFIFOQueue(queue_size, [tf.float32, tf.float32, tf.int32],
                                              shapes=[[n_input], [n_output], [1]])
        self.enqueue_train = self.queue_train.enqueue(
            [self.sample_placeholder_train, self.result_placeholder_train, self.idFile_placeholder_train])
        self.enqueue_test = self.queue_test.enqueue(
            [self.sample_placeholder_test, self.result_placeholder_test, self.idFile_placeholder_test])

        # https://github.com/tensorflow/tensorflow/issues/2514
        # https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/rmGu1HAyPw4
        # Use of a flag that changes the input queue to another one, this way the model can
        # be tested using the test queue when required.
        self.select_q = tf.placeholder(tf.int32, [])
        self.queue = tf.QueueBase.from_list(
            self.select_q, [self.queue_train, self.queue_test])

        # Find any file as the reggex is *
        self.files = find_files(data_dir, '*')
        if not self.files:
            raise ValueError("No data files found in '{}'.".format(data_dir))

        print("training files length: {}".format(len(self.files)))

        self.test_files = find_files(test_dir, '*')
        if not self.test_files:
            raise ValueError(
                "No test data files found in '{}'.".format(test_dir))

        print("test files length: {}".format(len(self.test_files)))

        # Split the data into test and train datasets
        # range = int(len(self.files) * test_percentage)
        self.test_dataset = self.test_files
        self.train_dataset = self.files

    def dequeue(self, num_elements):
        """ Function for dequeueing a mini-batch

        :param num_elements: int size of minibatch
        :return:
        """
        data, label, id = self.queue.dequeue_many(num_elements)

        return data, label, id

    def queue_switch(self):
        return self.select_q

    def thread_main(self, sess, id, n_thread, test):
        """ Thread function to be launched as many times as required for loading the data
        from several files into the Tensorflow's queue.

        :param sess: Tensorflow's session
        :param id: thread ID
        :param test: bool for choosing between the queue to feed the data, True for test queue
        :return: void
        """
        global epoch
        stop = False
        # Go through the dataset multiple times
        if test:
            files = self.test_dataset
        else:
            files = self.train_dataset

        # while tensorflows coordinator doesn't want to stop, continue.
        while not stop:

            epoch += 1
            if not test:
                print("Number of epochs: {}".format(epoch))
            randomized_files = randomize_files(files)

            '''
            file_partitions = []
            for index, i in enumerate(files):
                if (index)%(n_thread-1)+1 == id:
                    file_partitions.append(i)
            randomized_files = randomize_files(file_partitions)
            '''
            iterator = load_data_samples(randomized_files)

            for data, label, id_file in iterator:
                # update coordinator's state
                if self.coord.should_stop():
                    stop = True
                    break

                if test:  # in train range and test thread
                    sess.run(self.enqueue_test,
                             feed_dict={self.sample_placeholder_test: data,
                                        self.result_placeholder_test: label,
                                        self.idFile_placeholder_test: id_file})
                else:  # below the rage -> train
                    sess.run(self.enqueue_train,
                             feed_dict={self.sample_placeholder_train: data,
                                        self.result_placeholder_train: label,
                                        self.idFile_placeholder_train: id_file})

    def start_threads(self, sess, n_threads=2):
        """ Reader threads' launcher, uses the first thread for feeding into the test queue
        and the rest for feeding into the train queue.

        :param sess:
        :param n_threads:
        :return: void
        """
        for id in range(n_threads):
            if id == 0:
                thread = threading.Thread(
                    target=self.thread_main, args=(sess, id, n_threads, True))
            else:
                thread = threading.Thread(
                    target=self.thread_main, args=(sess, id, n_threads, False))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads

    # not used anymore
    def decompose_data(self, data):

        levels = 96
        CO2 = data[0]
        surface_temperature = data[1]
        air_temperature = []
        humidity = []
        for i in range(2, levels * 2 + 2):
            if (i % 2) == 0:  # even
                air_temperature.append(denormalize(data[i], meanT, stdT))
            else:
                humidity.append(denormalize(data[i], meanH, stdH))

        input_dic = {
            "surface_temperature": surface_temperature,
            "co2": CO2,
            "air_temperature": air_temperature,
            "humidity": humidity
        }

        return input_dic
