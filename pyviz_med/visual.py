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


class PyPatient:
    """Class for holding image and metadata for a patient."""

    def __init__(self, path, img_type='infer', overlay_path=None, *args, **kwargs):
        '''Initialize via reading the image and creating the xarray.'''

        np_img_list = self.get_img_list(path)
        np_overlay_list = self.get_img_list(overlay_path)

        self.n_img = len(np_img_list)
        self.n_overlay = len(np_overlay_list) if np_overlay_list else 0

        np_img_stack, np_overlay_stack = self.pad_and_stack_images(np_img_list, np_overlay_list)

        self.subject_id = kwargs.get('subject_id', 'no_id')
        self.label = kwargs.get('label', [f'image_{n}' for n in range(self.n_img)])
        if np_overlay_list is None:
            self.ds = xr.Dataset({'image': (['subject_id', 'label', 'z', 'y', 'x'],
                                            np_img_stack[np.newaxis, :])
                                 },
                                 coords={'subject_id': [self.subject_id],
                                         'label': self.label,
                                         'z': range(np_img_stack.shape[1]),
                                         'y': range(np_img_stack.shape[2])[::-1],
                                         'x': range(np_img_stack.shape[3])
                                        }
                                 )
        else:
            self.feature = kwargs.get('feature', [f'feature_{n}' for n in range(self.n_overlay)])
            self.ds = xr.Dataset({'image': (['subject_id', 'label', 'z', 'y', 'x'],
                                            np_img_stack[np.newaxis, :]),
                                  'overlay': (['subject_id', 'feature', 'z', 'y', 'x'],
                                              np_overlay_stack[np.newaxis, :])
                                 },
                                 coords={'name': [self.subject_id],
                                         'label': self.label,
                                         'feature': self.feature,
                                         'z': range(np_img_stack.shape[1]),
                                         'y': range(np_img_stack.shape[2])[::-1],
                                         'x': range(np_img_stack.shape[3])
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


    def get_img_list(self, path):
        if path is None:
            return None

        np_img_list = []
        if type(path) is list:
            for p in path:
                np_img_list.append(self.read_image(p[0], p[1]))
        else:
            np_img_list = [self.read_image(path[0], path[1])]

        return np_img_list


    def pad_and_stack_images(self, img_list, overlay_list=None):
        max_z = max_y = max_x = 0

        total_list = img_list
        if overlay_list is not None:
            total_list += overlay_list

        for img in total_list:
            max_z = max(img.shape[0], max_z)
            max_y = max(img.shape[1], max_y)
            max_x = max(img.shape[2], max_x)
        pad = np.zeros((max_z, max_y, max_x))

        for i,img in enumerate(img_list):
            pad_copy = pad.copy()
            pad_copy[:img.shape[0], :img.shape[1], :img.shape[2]]=img
            img_list[i] = pad_copy

        img_list = np.stack(img_list, axis=0)

        if overlay_list is not None:
            for i,overlay in enumerate(overlay_list):
                pad_copy = pad.copy()
                pad_copy[:overlay.shape[0], :overlay.shape[1], :overlay.shape[2]]=overlay
                overlay_list[i] = pad_copy

            overlay_list = np.stack(overlay_list, axis=0)

        return img_list, overlay_list


    def view(self, three_axis=False, default_size=300):
        # imopts = {'tools': ['hover'], 'width': 400, 'height': 400, 'cmap': 'gray'}
        imopts = {'tools': ['hover'], 'cmap': 'gray'}
        opts.defaults(
            opts.GridSpace(shared_xaxis=False, shared_yaxis=False,
                           fontsize={'title': 16, 'labels': 16, 'xticks': 12,
                                     'yticks': 12},
                           ),
            opts.Image(cmap='gray', tools=['hover'], xaxis=None,
                       yaxis=None),
        )
        hv_ds = hv.Dataset(self.ds)

        cslider = pn.widgets.RangeSlider(start=-3000, end=3000, value=(-800, 800), name='contrast')
        if three_axis:
            squish_height = int(max(default_size*(len(self.ds.z)/len(self.ds.x)), default_size/2))
            gridspace = hv.GridSpace(kdims=['plane', 'label'], label=f'{self.subject_id}')
            for mod in self.label:
                gridspace['axial', mod] = hv_ds.select(label=mod).to(
                    hv.Image, ['x', 'y'], groupby=['z'],
                    dynamic=True).opts(frame_width=default_size, frame_height=default_size).apply.opts(clim=cslider.param.value)
                gridspace['coronal', mod] = hv_ds.select(label=mod).to(
                    hv.Image, ['x', 'z'], groupby=['y'],
                    dynamic=True).opts(frame_width=default_size, frame_height=squish_height).apply.opts(clim=cslider.param.value)
                gridspace['sagittal', mod] = hv_ds.select(label=mod).to(
                    hv.Image, ['y', 'z'], groupby=['x'],
                    dynamic=True).opts(frame_width=default_size, frame_height=squish_height).apply.opts(clim=cslider.param.value)

        else:
            squish_height = int(max(default_size*(len(self.ds.z)/len(self.ds.x)), default_size/2))
            gridspace = hv.GridSpace(kdims=['label'], label=f'{self.subject_id}')
            for mod in self.label:
                gridspace[mod] = hv_ds.select(label=mod).to(
                    hv.Image, ['x', 'y'], groupby=['z'],
                    dynamic=True).opts(frame_width=default_size, frame_height=default_size).apply.opts(clim=cslider.param.value)

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
