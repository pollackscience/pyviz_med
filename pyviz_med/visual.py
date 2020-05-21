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
        np_img_list = []
        if type(path) is list:
            n_mod = len(path)
            for p in path:
                np_img_list.append(self.read_image(p, img_type))
        else:
            n_mod = 1
            np_img_list = [self.read_image(path, img_type)]

        max_z = max_y = max_x = 0
        for img in np_img_list:
            max_z = max(img.shape[0], max_z)
            max_y = max(img.shape[1], max_y)
            max_x = max(img.shape[2], max_x)
        pad = np.zeros((max_z, max_y, max_x))
        print(pad.shape)
        for i,img in enumerate(np_img_list):
            pad_copy = pad.copy()
            pad_copy[:img.shape[0], :img.shape[1], :img.shape[2]]=img
            np_img_list[i] = pad_copy

        np_img_list = np.stack(np_img_list, axis=0)

        self.subject = kwargs.get('subject', 'Unnamed')
        self.modality = kwargs.get('modality', ['Unknown']*n_mod)

        print(np_img_list.shape)
        self.ds = xr.Dataset({'image': (['subject', 'modality', 'z', 'y', 'x'],
                                        np_img_list[np.newaxis, :])
                             },
                             coords={'subject': [self.subject],
                                     'modality': self.modality,
                                     'z': range(np_img_list.shape[1]),
                                     'y': range(np_img_list.shape[2])[::-1],
                                     'x': range(np_img_list.shape[3])
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
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            print(dicom_names)
            dicom_names = sorted(dicom_names, key=lambda a: Path(a).stem[2:].zfill(3))
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
            reader.LoadPrivateTagsOn()  # Get DICOM Info
            image = reader.Execute()
            print('direction', image.GetDirection())
            print('origin', image.GetOrigin())
            print('spacing', image.GetSpacing())

        return sitk.GetArrayFromImage(image)


    def view(self, three_axis=False):
        imopts = {'tools': ['hover', 'lasso_select'], 'width': 400, 'height': 400, 'cmap': 'gray'}
        opts.defaults(
            opts.GridSpace(shared_xaxis=True, shared_yaxis=True,
                           fontsize={'title': 16, 'labels': 16, 'xticks': 12,
                                     'yticks': 12},
                           plot_size=300),
        )
        hv_ds = hv.Dataset(self.ds)

        cslider = pn.widgets.RangeSlider(start=-3000, end=3000, value=(-800, 800), name='contrast')
        if three_axis:
            gridspace = hv.GridSpace(kdims=['plane', 'modality'], label=f'{self.subject}')
            for mod in self.modality:
                gridspace['axial', mod] = hv_ds.select(modality=mod).to(
                    hv.Image, ['x', 'y'], groupby=['z'],
                    dynamic=True).opts(**imopts).apply.opts(clim=cslider.param.value)
                gridspace['coronal', mod] = hv_ds.select(modality=mod).to(
                    hv.Image, ['x', 'z'], groupby=['y'],
                    dynamic=True).opts(**imopts).apply.opts(clim=cslider.param.value)
                gridspace['sagittal', mod] = hv_ds.select(modality=mod).to(
                    hv.Image, ['y', 'z'], groupby=['x'],
                    dynamic=True).opts(**imopts).apply.opts(clim=cslider.param.value)

        else:
            gridspace = hv.GridSpace(kdims=['modality'], label=f'self.subject')
            for mod in self.modality:
                gridspace[mod] = hv_ds.select(modality=mod).to(
                    hv.Image, ['x', 'y'], groupby=['z'],
                    dynamic=True).opts(**imopts).apply.opts(clim=cslider.param.value)

        # hv_image = hv_ds.to(hv.Image, ['x', 'y'], groupby=['z', 'modality'], dynamic=True,
        #                     label=f'subject={self.subject}', group=f'modality={self.modality}').opts(**imopts)
        # if three_axis:
        #     hv_image_2 = hv_ds.to(hv.Image, ['x', 'z'], groupby=['y', 'modality'], dynamic=True).opts(**imopts)
        #     hv_image_3 = hv_ds.to(hv.Image, ['y', 'z'], groupby=['x', 'modality'], dynamic=True).opts(**imopts)
        #     img_list = [hv_image, hv_image_2, hv_image_3]
        # else:
        #     img_list = [hv_image]

        # Apply sliders
        # cslider = pn.widgets.RangeSlider(start=-3000, end=3000, value=(-800, 800), name='contrast')
        # for i in range(len(img_list)):
        #     img_list[i] = img_list[i].apply.opts(clim=cslider.param.value)

        # combo_image = reduce(lambda a,b : a+b,img_list)

        # Make widget panel
        #if len(self.ds.modality)>1:
        #    print(combo_image)
        #    layout = combo_image.grid('modality').cols(3)
        #else:
        #    layout = combo_image
        # pn_layout = pn.pane.HoloViews(layout)
        pn_layout = pn.pane.HoloViews(gridspace)
        wb = pn_layout.widget_box
        wb.append(cslider)
        return pn.Row(wb, pn_layout)
