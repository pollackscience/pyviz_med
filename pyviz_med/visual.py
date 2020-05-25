from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import SimpleITK as sitk
from bokeh.models import HoverTool
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
        np_img_stack, np_overlay_stack = self.pad_and_stack_images(np_img_list, np_overlay_list)

        self.n_img = np_img_stack.shape[0]
        self.n_overlay = np_overlay_stack.shape[0] if np_overlay_stack is not None else 0

        print(np_img_stack.shape)
        print(np_overlay_stack.shape)

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
                                  'overlay': (['subject_id', 'label', 'feature', 'z', 'y', 'x'],
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
            elif '.png' in str(path):
                img_type = 'png'
        elif img_type not in ['dicom', 'nifti', 'png']:
            raise IOError('Cannot infer image type, please specify "img_type"')

        if img_type == 'nifti':
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            image = reader.Execute()
            print('direction', image.GetDirection())
            print('origin', image.GetOrigin())
            print('spacing', image.GetSpacing())
            image = sitk.GetArrayFromImage(image)

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
            image = sitk.GetArrayFromImage(image)

        elif img_type == 'png':
            png_list = sorted(list(path.glob('*.png')))
            for i, png in enumerate(png_list):
                img_slice = sitk.GetArrayFromImage(sitk.ReadImage(str(png)))
                if i == 0:
                    image = np.zeros((len(png_list), img_slice.shape[0],
                                              img_slice.shape[1]), dtype=img_slice.dtype)
                image[i, :, :] = img_slice

        return image


    def get_img_list(self, path):
        if path is None:
            return None

        np_img_list = []
        if type(path) is list:
            for i in path:
                if type(i) is list:
                    np_img_list.append([])
                    for j in i:
                        np_img_list[-1].append(self.read_image(j[0], j[1]))
                else:
                    np_img_list.append(self.read_image(i[0], i[1]))
        else:
            np_img_list = [self.read_image(path[0], path[1])]

        return np_img_list


    def pad_and_stack_images(self, img_list, overlay_list=None):
        max_z = max_y = max_x = 0


        for img in img_list:
            max_z = max(img.shape[-3], max_z)
            max_y = max(img.shape[-2], max_y)
            max_x = max(img.shape[-1], max_x)
        if overlay_list is not None:
            n_features = self.get_n_features(overlay_list)
            print(n_features)
            for img in overlay_list:
                if type(img) is list:
                    for subimg in img:
                        max_z = max(subimg.shape[-3], max_z)
                        max_y = max(subimg.shape[-2], max_y)
                        max_x = max(subimg.shape[-1], max_x)
                else:
                    max_z = max(img.shape[-3], max_z)
                    max_y = max(img.shape[-2], max_y)
                    max_x = max(img.shape[-1], max_x)
        pad = np.zeros((max_z, max_y, max_x))

        for i,img in enumerate(img_list):
            pad_copy = pad.copy()
            pad_copy[:img.shape[0], :img.shape[1], :img.shape[2]]=img
            img_list[i] = pad_copy

        img_list = np.stack(img_list, axis=0)

        if overlay_list is not None:
            pad_overlay = np.zeros((n_features, max_z, max_y, max_x))
            for i,overlay in enumerate(overlay_list):
                if type(overlay) is list:
                    feat = 0
                    for j,sub_overlay in enumerate(overlay):
                        if sub_overlay.ndim == 3:
                            pad_copy[feat, :overlay.shape[0], :overlay.shape[1], :overlay.shape[2]]=overlay
                            feat += 1
                        elif overlay.ndim == 4:
                            pad_copy[feat:feat+overlay.shape[0], :overlay.shape[1], :overlay.shape[2], :overlay.shape[3]]=overlay
                            feat += overlay.shape[0]

                else:
                    pad_copy = pad_overlay.copy()
                    if overlay.ndim == 3:
                        pad_copy[0, :overlay.shape[0], :overlay.shape[1], :overlay.shape[2]]=overlay
                    elif overlay.ndim == 4:
                        pad_copy[0:overlay.shape[0], :overlay.shape[1], :overlay.shape[2], :overlay.shape[3]]=overlay
                overlay_list[i] = pad_copy

            overlay_list = np.stack(overlay_list, axis=0)

        return img_list, overlay_list

    def get_n_features(self, overlay_list):
        total_features = 0
        for i in overlay_list:
            sub_features = 0
            if type(i) is list:
                for j in i:
                    if len(j.shape) == 3:
                        sub_features += 1
                    elif len(j.shape == 4):
                        sub_features += j.shape[0]
            else:
                if len(i.shape) == 3:
                    sub_features += 1
                elif len(i.shape == 4):
                    sub_features += j.shape[0]
            total_features = max(total_features, sub_features)

        return total_features

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
            opts.Overlay(tools=['hover']),
            opts.NdOverlay(tools=['hover']),
        )
        hv_ds = hv.Dataset(self.ds)

        cslider = pn.widgets.RangeSlider(start=-3000, end=3000, value=(-800, 800), name='contrast')
        if 'overlay' in self.ds.data_vars:
            tooltips = [
                ('x', '@x'),
                ('y', '@y'),
                ('z', '@z'),
                ('image', '@image'),
                ('overlay', '@overlay')
            ]
            hover = HoverTool(tooltips=tooltips)
            overlay_max = self.ds.overlay.max()
            alpha_slider = pn.widgets.FloatSlider(start=0, end=1, value=0.7, name='overlay transparency')

            if three_axis:
                squish_height = int(max(default_size*(len(self.ds.z)/len(self.ds.x)), default_size/2))
                gridspace = hv.GridSpace(kdims=['plane', 'label'], label=f'{self.subject_id}')
                for mod in self.label:
                    gridspace['axial', mod] = hv_ds.select(label=mod).to(
                        hv.Image, ['x', 'y'], groupby=['z'], vdims='image',
                        dynamic=True).opts(frame_width=default_size, frame_height=default_size).apply.opts(clim=cslider.param.value)
                    gridspace['coronal', mod] = hv_ds.select(label=mod).to(
                        hv.Image, ['x', 'z'], groupby=['y'], vdims='image',
                        dynamic=True).opts(frame_width=default_size, frame_height=squish_height).apply.opts(clim=cslider.param.value)
                    gridspace['sagittal', mod] = hv_ds.select(label=mod).to(
                        hv.Image, ['y', 'z'], groupby=['x'], vdims='image',
                        dynamic=True).opts(frame_width=default_size, frame_height=squish_height).apply.opts(clim=cslider.param.value)

                    gridspace['axial', mod] *= hv_ds.select(label=mod).to(
                        hv.Image, ['x', 'y'], groupby=['z', 'feature'], vdims='overlay',
                        dynamic=True).opts(
                            cmap='glasbey_hv', clipping_colors={'min': 'transparent'},
                        ).redim.range(overlay=(0.1, overlay_max)).apply.opts(
                            alpha=alpha_slider.param.value)
                    gridspace['coronal', mod] *= hv_ds.select(label=mod).to(
                        hv.Image, ['x', 'z'], groupby=['y', 'feature'], vdims='overlay',
                        dynamic=True).opts(
                            cmap='glasbey_hv', clipping_colors={'min': 'transparent'},
                        ).redim.range(overlay=(0.1, overlay_max)).apply.opts(
                            alpha=alpha_slider.param.value)
                    gridspace['sagittal', mod] *= hv_ds.select(label=mod).to(
                        hv.Image, ['y', 'z'], groupby=['x', 'feature'], vdims='overlay',
                        dynamic=True).opts(
                            cmap='glasbey_hv', clipping_colors={'min': 'transparent'},
                        ).redim.range(overlay=(0.1, overlay_max)).apply.opts(
                            alpha=alpha_slider.param.value)

            else:
                squish_height = int(max(default_size*(len(self.ds.z)/len(self.ds.x)), default_size/2))
                gridspace = hv.GridSpace(kdims=['label'], label=f'{self.subject_id}')
                for mod in self.label:
                    gridspace[mod] = hv_ds.select(label=mod).to(
                        hv.Image, ['x', 'y'], groupby=['z'], vdims='image',
                        dynamic=True).opts(frame_width=default_size, frame_height=default_size,
                                           ).apply.opts(clim=cslider.param.value)
                    gridspace[mod] *= hv_ds.select(label=mod).to(
                        hv.Image, ['x', 'y'], groupby=['z', 'feature'], vdims='overlay',
                        dynamic=True).opts(
                            cmap='glasbey_hv', clipping_colors={'min': 'transparent'},
                        ).redim.range(overlay=(0.1, overlay_max)).apply.opts(
                            alpha=alpha_slider.param.value)
                    gridspace[mod] = gridspace[mod].opts(tools=['hover'])
                    print(gridspace[mod])
        else:
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

        pn_layout = pn.pane.HoloViews(gridspace)
        wb = pn_layout.widget_box
        wb.append(cslider)
        if 'overlay' in self.ds.data_vars:
            wb.append(alpha_slider)
        return pn.Row(wb, pn_layout)
