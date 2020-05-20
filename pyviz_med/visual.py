from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import SimpleITK as sitk
import panel as pn
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize

hv.extension('bokeh')


class MedImage:
    """Class for holding a medical image."""

    def __init__(self, path, img_type='infer', *args, **kwargs):
        '''Initialize via reading the image and creating the xarray.'''
        self.read_image(path, img_type)

        self.subject = kwargs.get('subject', 'Unnamed')
        self.modality = kwargs.get('modality', 'Unknown')

        print(self.img_np.shape)
        self.ds = xr.Dataset({'image': (['subject', 'modality', 'z', 'y', 'x'],
                                        self.img_np[np.newaxis, np.newaxis, :])
                              },

                             coords={'subject': [self.subject],
                                     'modality': [self.modality],
                                     'z': range(self.img_np.shape[0]),
                                     'y': range(self.img_np.shape[1]),
                                     'x': range(self.img_np.shape[2])
                                    }
                             )


    def read_image(self, path, img_type):
        '''Read image from path, and store image object'''
        # Clean args
        path = Path(path)
        img_type = img_type.lower()

        if img_type == 'infer':
            if '.dcm' in str(path):
                img_type = 'dicom'
            elif '.nii' in str(path):
                img_type = 'nifti'
        elif img_type not in ['dicom', 'nifti']:
            raise IOError('Cannot infer image type, please specify "img_type"')

        if img_type == 'nifti':
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            image = reader.Execute()
            print('direction', image.GetDirection())
            print('origin', image.GetOrigin())
            print('spacing', image.GetSpacing())

        elif img_type == 'dicom':
            reader = sitk.ImageSeriesReader()
            print(str(path))
            dicom_names = reader.GetGDCMSeriesFileNames(str(path)[:-5])
            print(dicom_names)
            dicom_names = sorted(dicom_names, key=lambda a: Path(a).stem[2:].zfill(3))
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
            reader.LoadPrivateTagsOn()  # Get DICOM Info
            image = reader.Execute()
            print('direction', image.GetDirection())
            print('origin', image.GetOrigin())
            print('spacing', image.GetSpacing())

        self.img_np = sitk.GetArrayFromImage(image)


    def view(self):
        hv_ds = hv.Dataset(self.ds)
        return hv_ds.to(hv.Image, ['x', 'y'], groupby='z', dynamic=True)



def patient_series_viewer(path, patient, img_type='DICOM', info=''):
    '''Similar to pybreast viewer, but must natively handle a mix of 2d, 3d, scalar, and vector'''

    imopts = {'tools': ['hover', 'lasso_select'], 'width': 300, 'height': 300, 'cmap': 'viridis'}
    full_path = Path(path, patient)

    if img_type == 'NIFTI':
        img_folders = sorted(list(full_path.glob('*.nii')), key=lambda a: a.stem.split('_'))
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
    elif img_type == 'DICOM':
        img_folders = sorted(list(full_path.iterdir()), key=lambda a: int(a.stem[2:]))
        reader = sitk.ImageSeriesReader()
    elif img_type == 'DICOM_CHAOS':
        img_folders = [Path(full_path, 'T1DUAL/DICOM_anon/InPhase'),
                       Path(full_path, 'T1DUAL/DICOM_anon/OutPhase'),
                       Path(full_path, 'T2SPIR/DICOM_anon')]
        reader = sitk.ImageSeriesReader()
    elif img_type == 'DICOM_CHAOS_CT':
        img_folders = [Path(full_path, 'DICOM_anon')]
        reader = sitk.ImageSeriesReader()

    else:
        raise KeyError(f'img_type must be one of ["DICOM", "NIFTI"], got {img_type}')

    hv_images = []
    for img_files in img_folders:
        print(img_files)
        hvds_list = []
        if 'DICOM' in img_type:
            dicom_names = reader.GetGDCMSeriesFileNames(str(img_files))
            dicom_names = sorted(dicom_names, key=lambda a: Path(a).stem[2:].zfill(3))
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
            reader.LoadPrivateTagsOn()  # Get DICOM Info
            image = reader.Execute()
            print('direction', image.GetDirection())
            print('origin', image.GetOrigin())
            print('spacing', image.GetSpacing())
            desc = reader.GetMetaData(0, '0008|103e').strip().encode('utf-8', 'ignore').decode()
            pid = reader.GetMetaData(0, '0010|0010').strip()
            desc = ' '.join([img_files.stem, desc, pid])
            # if image.GetOrigin()[-1] > 0:
            #     image = image[:, :, ::-1]
        elif img_type == 'NIFTI':
            reader.SetFileName(str(img_files))
            desc = ' '.join(img_files.parts[-2:])
            image = reader.Execute()
            print('direction', image.GetDirection())
            print('origin', image.GetOrigin())
            print('spacing', image.GetSpacing())

        npimg = sitk.GetArrayFromImage(image)
        print(npimg.shape)
        if npimg.shape[0] == 1:
            hv_images.append(hv.Image(npimg[0, :], label=desc).opts(**imopts))
        elif npimg.shape[-1] > 3:
            hvds_list.append(hv.Dataset(
                (np.arange(npimg.shape[2]), np.arange(npimg.shape[1])[::-1],
                 np.arange(npimg.shape[0]),
                 npimg), [f'x{desc}', f'y{desc}', f'z{desc}'],
                f'MRI{desc}'))
            print(hvds_list[-1])
            hv_images.append(hvds_list[-1].to(hv.Image, [f'x{desc}', f'y{desc}'],
                                              groupby=[f'z{desc}'],
                                              dynamic=True, label=desc).opts(**imopts,
                                                                             invert_yaxis=False))
        else:
            hv_images.append(hv.Image(npimg[0, :], label=desc).opts(**imopts))
        print()
    return hv.Layout(hv_images).opts(shared_axes=False, merge_tools=False, normalize=False,
                                     title=' '.join([patient, info])).cols(3)

