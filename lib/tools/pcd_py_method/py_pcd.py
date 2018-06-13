import os
import re
import struct
import copy
import numpy as np
import warnings
import socket
if socket.gethostname()=="hexindong":
    from tools.data_visualize import pcd_vispy, vispy_init
    vispy_init() # must before 'import lzf'
import lzf

HAS_SENSOR_MSGS = True
try:
    from sensor_msgs.msg import PointField
    import numpy_pc2  # needs sensor_msgs
except ImportError:
    HAS_SENSOR_MSGS = False

class point_cloud(object):
    def __init__(self, metadata, pc_data):
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.pc_data = pc_data
        self.check_sanity()

    def get_metadata(self):
        """ returns copy of metadata """
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    def check_sanity(self):
        # pdb.set_trace()
        md = self.get_metadata()
        assert(self._metadata_is_consistent(md))
        assert(len(self.pc_data) == self.points)
        assert(self.width*self.height == self.points)
        assert(len(self.fields) == len(self.count))
        assert(len(self.fields) == len(self.type))

    def save(self, fname):
        self.save_pcd(fname, 'ascii')

    def save_pcd(self, fname, compression=None, **kwargs):
        if 'data_compression' in kwargs:
            warnings.warn('data_compression keyword is deprecated for'
                          ' compression')
            compression = kwargs['data_compression']
        with open(fname, 'w') as f:
            self.point_cloud_to_fileobj(self, f, compression)

    def to_msg(self):
        if not HAS_SENSOR_MSGS:
            raise NotImplementedError('ROS sensor_msgs not found')
        # TO-DO is there some metadata we want to attach?
        return numpy_pc2.array_to_pointcloud2(self.pc_data)

    @staticmethod
    def point_cloud_to_fileobj(pc, fileobj, data_compression=None):
        """ write pointcloud as .pcd to fileobj.
        if data_compression is not None it overrides pc.data.
        """

        def write_header(_metadata, rename_padding=False):
            """ given metadata as dictionary return a string header.
            """
            template = """\
        VERSION {version}
        FIELDS {fields}
        SIZE {size}
        TYPE {type}
        COUNT {count}
        WIDTH {width}
        HEIGHT {height}
        VIEWPOINT {viewpoint}
        POINTS {points}
        DATA {data}
        """
            str_metadata = _metadata.copy()

            if not rename_padding:
                str_metadata['fields'] = ' '.join(_metadata['fields'])
            else:
                new_fields = []
                for f in _metadata['fields']:
                    if f == '_':
                        new_fields.append('padding')
                    else:
                        new_fields.append(f)
                str_metadata['fields'] = ' '.join(new_fields)
            str_metadata['size'] = ' '.join(map(str, _metadata['size']))
            str_metadata['type'] = ' '.join(_metadata['type'])
            str_metadata['count'] = ' '.join(map(str, _metadata['count']))
            str_metadata['width'] = str(_metadata['width'])
            str_metadata['height'] = str(_metadata['height'])
            str_metadata['viewpoint'] = ' '.join(map(str, _metadata['viewpoint']))
            str_metadata['points'] = str(_metadata['points'])
            tmpl = template.format(**str_metadata)
            return tmpl

        def build_ascii_fmtstr(pc_):
            """ make a format string for printing to ascii, using fields
            %.8f minimum for rgb
            %.10f for more general use?
            """
            fmtstr = []
            for t, cnt in zip(pc_.type, pc_.count):
                if t == 'F':
                    fmtstr.extend(['%.10f'] * cnt)
                elif t == 'I':
                    fmtstr.extend(['%d'] * cnt)
                elif t == 'U':
                    fmtstr.extend(['%u'] * cnt)
                else:
                    raise ValueError("don't know about type %s" % t)
            return fmtstr

        metadata = pc.get_metadata()
        if data_compression is not None:
            data_compression = data_compression.lower()
            assert (data_compression in ('ascii', 'binary', 'binary_compressed'))
            metadata['data'] = data_compression

        header = write_header(metadata)
        fileobj.write(header)
        if metadata['data'].lower() == 'ascii':
            fmtstr = build_ascii_fmtstr(pc)
            np.savetxt(fileobj, pc.pc_data, fmt=fmtstr)
        elif metadata['data'].lower() == 'binary':
            fileobj.write(pc.pc_data.tostring('C'))
        elif metadata['data'].lower() == 'binary_compressed':
            # TO-DO
            # a '_' field is ignored by pcl and breakes compressed point clouds.
            # changing '_' to '_padding' or other name fixes this.
            # admittedly padding shouldn't be compressed in the first place
            # reorder to column-by-column
            uncompressed_lst = []
            for fieldname in pc.pc_data.dtype.names:
                column = np.ascontiguousarray(pc.pc_data[fieldname]).tostring('C')
                uncompressed_lst.append(column)
            uncompressed = ''.join(uncompressed_lst)
            uncompressed_size = len(uncompressed)
            # print("uncompressed_size = %r"%(uncompressed_size))
            buf = lzf.compress(uncompressed)
            if buf is None:
                # compression didn't shrink the file
                # TO-DO what do to do in this case when reading?
                buf = uncompressed
                compressed_size = uncompressed_size
            else:
                compressed_size = len(buf)
            fmt = 'II'
            fileobj.write(struct.pack(fmt, compressed_size, uncompressed_size))
            fileobj.write(buf)
        else:
            raise ValueError('unknown DATA type')
        # we can't close because if it's stringio buf then we can't get value after

    @staticmethod
    def _metadata_is_consistent(metadata):
        """ sanity check for metadata. just some basic checks.
        """
        checks = []
        required = ('version', 'fields', 'size', 'width', 'height', 'points',
                    'viewpoint', 'data')
        for f in required:
            if f not in metadata:
                print('%s required' % f)
        checks.append((lambda m: all([k in m for k in required]),
                       'missing field'))
        checks.append((lambda m: len(m['type']) == len(m['count']) ==
                                 len(m['fields']),
                       'length of type, count and fields must be equal'))
        checks.append((lambda m: m['height'] > 0,
                       'height must be greater than 0'))
        checks.append((lambda m: m['width'] > 0,
                       'width must be greater than 0'))
        checks.append((lambda m: m['points'] > 0,
                       'points must be greater than 0'))
        checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                                                       'binary_compressed'),
                       'unknown data type:'
                       'should be ascii/binary/binary_compressed'))
        ok = True
        for check, msg in checks:
            if not check(metadata):
                print('error:', msg)
                ok = False
        return ok

    @staticmethod
    def from_path(fname):
        """ parse pointcloud coming from file object f
        """
        def parse_header(lines):
            metadata = {}
            for ln in lines:
                if ln.startswith('#') or len(ln) < 2:
                    continue
                match = re.match('(\w+)\s+([\w\s\.]+)', ln)
                if not match:
                    warnings.warn("warning: can't understand line: %s" % ln)
                    continue
                key, value = match.group(1).lower(), match.group(2)
                if key == 'version':
                    metadata[key] = value
                elif key in ('fields', 'type'):
                    metadata[key] = value.split()
                elif key in ('size', 'count'):
                    metadata[key] = map(int, value.split())
                elif key in ('width', 'height', 'points'):
                    metadata[key] = int(value)
                elif key == 'viewpoint':
                    metadata[key] = map(float, value.split())
                elif key == 'data':
                    metadata[key] = value.strip().lower()
                # TO-DO apparently count is not required?
            # add some reasonable defaults
            if 'count' not in metadata:
                metadata['count'] = [1] * len(metadata['fields'])
            if 'viewpoint' not in metadata:
                metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            if 'version' not in metadata:
                metadata['version'] = '.7'
            return metadata

        def _build_dtype(metadata_):
            """ build numpy structured array dtype from pcl metadata.
            note that fields with count > 1 are 'flattened' by creating multiple
            single-count fields.
            TO-DO: allow 'proper' multi-count fields.
            """
            fieldnames = []
            typenames = []
            numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                                       (np.dtype('float64'), ('F', 8)),
                                       (np.dtype('uint8'), ('U', 1)),
                                       (np.dtype('uint16'), ('U', 2)),
                                       (np.dtype('uint32'), ('U', 4)),
                                       (np.dtype('uint64'), ('U', 8)),
                                       (np.dtype('int16'), ('I', 2)),
                                       (np.dtype('int32'), ('I', 4)),
                                       (np.dtype('int64'), ('I', 8))]
            pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)

            for f, c, t, s in zip(metadata_['fields'],
                                  metadata_['count'],
                                  metadata_['type'],
                                  metadata_['size']):
                np_type = pcd_type_to_numpy_type[(t, s)]
                if c == 1:
                    fieldnames.append(f)
                    typenames.append(np_type)
                else:
                    fieldnames.extend(['%s_%04d' % (f, i) for i in xrange(c)])
                    typenames.extend([np_type] * c)
            dtype = np.dtype(zip(fieldnames, typenames))
            return dtype

        def parse_binary_pc_data(f, dtype, metadata):
            rowstep = metadata['points'] * dtype.itemsize
            # for some reason pcl adds empty space at the end of files
            buf = f.read(rowstep)
            return np.fromstring(buf, dtype=dtype)

        def parse_binary_compressed_pc_data(f, dtype, metadata):
            # compressed size of data (uint32)
            # uncompressed size of data (uint32)
            # compressed data
            # junk
            fmt = 'II'
            compressed_size, uncompressed_size = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
            compressed_data = f.read(compressed_size)
            # (compressed > uncompressed)
            # should we read buf as raw binary?
            buf = lzf.decompress(compressed_data, uncompressed_size)
            if len(buf) != uncompressed_size:
                raise Exception('Error decompressing data')
            # the data is stored field-by-field
            pcs_data = np.zeros(metadata['width'], dtype=dtype)
            ix = 0
            for dti in range(len(dtype)):
                dt = dtype[dti]
                bytess = dt.itemsize * metadata['width']
                column = np.fromstring(buf[ix:(ix + bytess)], dt)
                pcs_data[dtype.names[dti]] = column
                ix += bytess
            return pcs_data

        with open(fname, 'rb') as f:
            header = []
            while True:
                ln = f.readline().strip()
                header.append(ln)
                if ln.startswith('DATA'):
                    metadata = parse_header(header)
                    dtype = _build_dtype(metadata)
                    break
            if metadata['data'] == 'ascii':
                pc_data = np.loadtxt(f, dtype=dtype, delimiter=' ')
                pc_data.dtype = np.float32
                pc_data = pc_data.reshape(-1, 4)
            elif metadata['data'] == 'binary':
                pc_data = parse_binary_pc_data(f, dtype, metadata)
            elif metadata['data'] == 'binary_compressed':
                pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
            else:
                print('File->py_pcd.py: DATA field is not "ascii",maybe "binary" or "binary_compressed", try to add method for both')
                return 'CODE: 0x123'
            pc = point_cloud(metadata, pc_data)
        return pc


def generate_pcd(path,save_path):
    fileindex = sorted(os.listdir(path))
    for idx_, f_name in enumerate(fileindex):
        pc_data = np.fromfile(os.path.join(path, f_name), dtype=np.float32).reshape(-1, 4)
        cnt = pc_data.shape[0]
        metadata = dict({'count': [1, 1, 1, 1],
                         'data': 'ascii',
                         'fields': ['x', 'y', 'z', 'intensity'],
                         'height': 1,
                         'points': cnt,
                         'size': [4, 4, 4, 4],
                         'type': ['F', 'F', 'F', 'F'],
                         'version': '0.7',
                         'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                         'width': cnt,
                         })
        pointcloud = point_cloud(metadata, pc_data)
        pcd_name = os.path.join(save_path, str(idx_).zfill(6) + '.pcd')
        pointcloud.save(pcd_name)
        print 'save file: {}'.format(pcd_name)

def show_pcd(dataPath,box=None):
    fileindex = sorted(os.listdir(dataPath))
    for File in fileindex:
        pc = point_cloud.from_path(os.path.join(dataPath, File))
        if box is None:
            pcd_vispy(pc.pc_data)
        else:
            pcd_vispy(pc.pc_data,boxes=box)

if __name__=='__main__':
    # save_path = '/home/hexindong/Desktop/jj_data/pcd_file'
    # data_path = '/home/hexindong/Desktop/jj_data/bin_file'
    # generate_pcd(data_path,save_path)
    # print 'Convert the file done ! '

    pcd_file_path ='/home/hexindong/DATASET/DATA_BOXES/STI_BOX/pcd_car'
    box = {'center':np.array([0,0,0]).reshape(-1,3),
           'size':np.array([4,4,2]).reshape(-1,3),
           'cls_rpn':np.array([4]).reshape(-1,1),
           'score':np.array([1.0]).reshape(-1,1),
           'cls_cube':np.array([1.0]).reshape(-1,1),
           'yaw':np.array([0.0]).reshape(-1,1)
           }
    show_pcd(pcd_file_path,box)

