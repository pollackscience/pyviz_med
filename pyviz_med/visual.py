from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import SimpleITK as sitk
from bokeh.models import HoverTool
import panel as pn
import param
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from bokeh.models import FuncTickFormatter

# hv.extension('bokeh')


class PyPatient:
    """Class for holding image and metadata for a patient."""

    def __init__(self, path, img_type='infer', overlay_path=None, from_xarray=False, *args, **kwargs):
        '''Initialize via reading the image and creating the xarray.'''

        self.verbose = kwargs.get('verbose', False)
        # Initial trimming of path arg
        if type(path) is dict:
            self.label = list(path.keys())
            path = [path[key] for key in self.label]
        if type(overlay_path) is dict:
            self.overlay_label = list(overlay_path.keys())
            overlay_path = [overlay_path[key] for key in self.overlay_label]

        np_img_list, img_metadata = self.get_img_list(path)
        if self.verbose:
            print(img_metadata, len(np_img_list))
        np_overlay_list, _ = self.get_overlay_list(overlay_path)
        self.orient_images(np_img_list, img_metadata)
        self.orient_overlays(np_overlay_list, img_metadata)

        np_img_stack, np_overlay_stack = self.pad_and_stack_images(np_img_list, np_overlay_list)

        self.n_img = np_img_stack.shape[0]
        self.n_overlay = np_overlay_stack.shape[1] if np_overlay_stack is not None else 0

        self.subject_id = kwargs.get('subject_id', 'NO_ID')
        if not hasattr(self, 'label'):
            self.label = kwargs.get('label', [f'image_{n}' for n in range(self.n_img)])
        if type(self.label) is not list:
            self.label = [self.label]
        if np_overlay_list is None:
            if self.verbose:
                print(img_metadata)
                print(img_metadata['spacing'])
                print(img_metadata['origin'])
                print(img_metadata['direction'])
            self.ds = xr.Dataset({'image': (['subject_id', 'label', 'z', 'y', 'x'],
                                            np_img_stack[np.newaxis, :]),
                                  'spacing': (['subject_id', 'label', 'img_dims'],
                                              [img_metadata['spacing']]),
                                  'origin': (['subject_id', 'label', 'img_dims'],
                                              [img_metadata['origin']]),
                                 },
                                 coords={'subject_id': [self.subject_id],
                                         'label': self.label,
                                         'z': range(np_img_stack.shape[1]),
                                         'y': range(np_img_stack.shape[2]),
                                         'x': range(np_img_stack.shape[3]),
                                         'img_dims': range(3),
                                        }
                                 )
        else:
            if not hasattr(self, 'overlay_label'):
                self.overlay_label = kwargs.get('overlay_label', [f'overlay_label_{n}' for n in
                                                                  range(self.n_overlay)])
            self.ds = xr.Dataset({'image': (['subject_id', 'label', 'z', 'y', 'x'],
                                            np_img_stack[np.newaxis, :]),
                                  'overlay': (['subject_id', 'label', 'overlay_label', 'z', 'y', 'x'],
                                              np_overlay_stack[np.newaxis, :]),
                                  'spacing': (['subject_id', 'label', 'img_dims'],
                                              [img_metadata['spacing']]),
                                  'origin': (['subject_id', 'label', 'img_dims'],
                                              [img_metadata['origin']]),
                                 },
                                 coords={'subject_id': [self.subject_id],
                                         'label': self.label,
                                         'overlay_label': self.overlay_label,
                                         'z': range(np_img_stack.shape[1]),
                                         'y': range(np_img_stack.shape[2]),
                                         'x': range(np_img_stack.shape[3]),
                                         'img_dims': range(3),
                                        }
                                 )
    def orient_images(self, np_img_list, img_metadata, bad=False):
        if self.verbose:
            print(img_metadata)
        if np_img_list is None:
            return None
        for i in range(len(np_img_list)):
            if type(np_img_list[i]) is list:
                for j in range(len(np_img_list[i])):
                    if self.verbose:
                        print(img_metadata['direction'][i])
                    if img_metadata['direction'][i][0] < 0:
                        np_img_list[i][j] = np.flip(np_img_list[i][j], axis=2)
                    if img_metadata['direction'][i][4] > 0:
                        np_img_list[i][j] = np.flip(np_img_list[i][j], axis=1)
                    if img_metadata['direction'][i][8] < 0:
                        np_img_list[i][j] = np.flip(np_img_list[i][j], axis=0)

            else:
                if self.verbose:
                    print(img_metadata['direction'][i])
                if img_metadata['direction'][i][0] < 0:
                    np_img_list[i] = np.flip(np_img_list[i], axis=2)
                if bad:
                    if img_metadata['direction'][i][4] < 0:
                        np_img_list[i] = np.flip(np_img_list[i], axis=1)
                else:
                    if img_metadata['direction'][i][4] > 0:
                        np_img_list[i] = np.flip(np_img_list[i], axis=1)
                if img_metadata['direction'][i][8] < 0:
                    np_img_list[i] = np.flip(np_img_list[i], axis=0)


    def orient_overlays(self, np_overlay_list, img_metadata, bad=False):
        if self.verbose:
            print(img_metadata)
        if np_overlay_list is None:
            return None

        for i in range(len(np_overlay_list)):
            # loop over overlays (overlay_labels)
            if type(np_overlay_list[i]) is list:
                # loop over images (labels)
                for j in range(len(np_overlay_list[i])):
                    if self.verbose:
                        print(img_metadata['direction'][j])
                    if img_metadata['direction'][j][0] < 0:
                        np_overlay_list[i][j] = np.flip(np_overlay_list[i][j], axis=2)
                    if img_metadata['direction'][j][4] > 0:
                        np_overlay_list[i][j] = np.flip(np_overlay_list[i][j], axis=1)
                    if img_metadata['direction'][j][8] < 0:
                        np_overlay_list[i][j] = np.flip(np_overlay_list[i][j], axis=0)

            else:
                if self.verbose:
                    print(img_metadata['direction'][0])
                if img_metadata['direction'][0][0] < 0:
                    np_overlay_list[i] = np.flip(np_overlay_list[i], axis=2)
                if bad:
                    if img_metadata['direction'][0][4] < 0:
                        np_overlay_list[i] = np.flip(np_overlay_list[i], axis=1)
                else:
                    if img_metadata['direction'][0][4] > 0:
                        np_overlay_list[i] = np.flip(np_overlay_list[i], axis=1)
                if img_metadata['direction'][0][8] < 0:
                    np_overlay_list[i] = np.flip(np_overlay_list[i], axis=0)


    def get_file_type(self, path):
        '''Automatically determine file type based on path name.'''
        pass

    def parse_path(self, path):
        '''Return list of files that match path specification.'''

        path = Path(path)

        if path.is_file():
            return str(path)
        elif path.is_dir():
            return [str(p) for p in sorted(path.glob('*'))]
        elif '*' in path.name:
            return [str(p) for p in sorted(path.parents[0].glob(path.name))]
        else:
            raise ValueError('Cannot parse path: {path}')

        return path

    def read_image(self, path, img_type=None):
        '''Read image from path, and store image object'''
        # Clean args
        path = Path(path)
        if img_type is None:
            path = self.parse_path(path)
        else:
            img_type = img_type.lower()

        if img_type is None:
            image = sitk.ReadImage(path)
            direction = image.GetDirection()
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            image = sitk.GetArrayFromImage(image)
            if len(image.shape) == 4:
                image = image[0:1, :]

        elif img_type == 'nifti':
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            image = reader.Execute()
            direction = image.GetDirection()
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            image = sitk.GetArrayFromImage(image)
            if len(image.shape) == 4:
                image = image[0:1, :]

        elif img_type == 'dicom':
            reader = sitk.ImageSeriesReader()
            if self.verbose:
                print(str(path))
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            if self.verbose:
                print(dicom_names)
            dicom_names = sorted(dicom_names, key=lambda a: Path(a).stem[2:].zfill(3))
            reader.SetFileNames(dicom_names)
            # reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
            # reader.LoadPrivateTagsOn()  # Get DICOM Info
            image = reader.Execute()
            direction = image.GetDirection()
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            image = sitk.GetArrayFromImage(image)

        elif img_type == 'png':
            png_list = sorted(list(path.glob('*.png')))
            for i, png in enumerate(png_list):
                img_slice = sitk.GetArrayFromImage(sitk.ReadImage(str(png)))
                if i == 0:
                    image = np.zeros((len(png_list), img_slice.shape[0],
                                              img_slice.shape[1]), dtype=img_slice.dtype)
                image[i, :, :] = img_slice
            direction = None
            origin = None
            spacing = None

        return image, {'spacing': spacing, 'origin': origin, 'direction': direction}


    def get_img_list(self, path, get_metadata=True):
        '''Return a list of images in numpy format (and metadata, optional).
        Possible path configs:
        '''
        if path is None:
            return None, None

        np_img_list = []
        meta_data_lists = {'direction':[], 'origin':[], 'spacing':[]}

        if type(path) is str:
            img, meta_data = self.read_image(path)
            np_img_list.append(img)
            for key in meta_data.keys():
                if self.verbose:
                    print(key)
                meta_data_lists[key].append(meta_data[key])
        elif type(path) is list:
            for i in path:
                img, meta_data = self.read_image(i)
                np_img_list.append(img)
                for key in meta_data.keys():
                    if self.verbose:
                        print(key)
                        print(meta_data_lists[key])
                    meta_data_lists[key].append(meta_data[key])

        return np_img_list, meta_data_lists

    def get_overlay_list(self, path, get_metadata=True):
        '''Return a list of overlays in numpy format (and metadata, optional).
        Possible configs: path, list(path), list(path, (lab1, lab2, ...))
        '''
        if path is None:
            return None, None

        np_overlay_list = []
        meta_data_lists = {'direction':[], 'origin':[], 'spacing':[]}

        if type(path) is str:
            img, meta_data = self.read_image(path)
            np_overlay_list.append(img)
            for key in meta_data.keys():
                if self.verbose:
                    print(key)
                meta_data_lists[key].append(meta_data[key])

        elif type(path) is list:
        #     if len(path) == 1:
        #         path = path[0]
            for i in path:
                if type(i) is tuple:
                    pass
                elif type(i) is list:
                    np_overlay_list.append([])
                    for j in i:
                        img, meta_data = self.read_image(j)
                        np_overlay_list[-1].append(img)
                        # for key in meta_data.keys():
                        #     if self.verbose:
                        #         print(key)
                        #         print(meta_data_lists[key])
                        #     meta_data_lists[key].append(meta_data[key])
                else:
                    img, meta_data = self.read_image(i)
                    np_overlay_list.append(img)
                    for key in meta_data.keys():
                        if self.verbose:
                            print(key)
                            print(meta_data_lists[key])
                        meta_data_lists[key].append(meta_data[key])

        return np_overlay_list, meta_data_lists


    def pad_and_stack_images(self, img_list, overlay_list=None):
        n_imgs = len(img_list)
        if self.verbose:
            print('len img_list', n_imgs)
        max_z = max_y = max_x = 0

        for img in img_list:
            max_z = max(img.shape[-3], max_z)
            max_y = max(img.shape[-2], max_y)
            max_x = max(img.shape[-1], max_x)
        if overlay_list is not None:
            n_overlay_labels = self.get_n_overlay_labels(overlay_list)
            if self.verbose:
                print('n_overlay_labels', n_overlay_labels)
                print(overlay_list)
            for img in overlay_list:
                if type(img) is list:
                    for subimg in img:
                        max_z = max(subimg.shape[-3], max_z)
                        max_y = max(subimg.shape[-2], max_y)
                        max_x = max(subimg.shape[-1], max_x)
                else:
                    if self.verbose:
                        print('img.shape', img.shape)
                    max_z = max(img.shape[-3], max_z)
                    max_y = max(img.shape[-2], max_y)
                    max_x = max(img.shape[-1], max_x)
        pad = np.zeros((max_z, max_y, max_x))

        for i,img in enumerate(img_list):
            pad_copy = pad.copy()
            pad_copy[:img.shape[0], :img.shape[1], :img.shape[2]]=img
            img_list[i] = pad_copy

        img_list = np.stack(img_list, axis=0)

        padded_overlay = None
        if overlay_list is not None:
            padded_overlay = np.zeros((n_imgs, n_overlay_labels, max_z, max_y, max_x))
            # print(padded_overlay.shape)
            # Loop over overlays
            feat = 0
            for i,overlay in enumerate(overlay_list):
                if type(overlay) is list:
                    # Loop over images
                    for j,sub_overlay in enumerate(overlay):
                        if sub_overlay.ndim == 3:
                            # print(sub_overlay.shape)
                            # print(j, feat)
                            padded_overlay[j, feat, :sub_overlay.shape[0], :sub_overlay.shape[1], :sub_overlay.shape[2]]=sub_overlay
                        elif sub_overlay.ndim == 4:
                            padded_overlay[j, feat:feat+sub_overlay.shape[0], :sub_overlay.shape[1],
                                           :sub_overlay.shape[2], :sub_overlay.shape[3]]=sub_overlay
                    if sub_overlay.ndim == 3:
                        feat += 1
                    elif sub_overlay.ndim == 4:
                        feat += sub_overlay.shape[0]

                else:
                    if overlay.ndim == 3:
                        padded_overlay[0, feat, :overlay.shape[0], :overlay.shape[1], :overlay.shape[2]]=overlay
                        feat += 1
                    elif overlay.ndim == 4:
                        padded_overlay[0, feat:feat+overlay.shape[0], :overlay.shape[1], :overlay.shape[2], :overlay.shape[3]]=overlay
                        feat += overlay.shape[0]

        return img_list, padded_overlay

    def get_n_overlay_labels(self, overlay_list):
        return len(overlay_list)
        # n_overlay_labels = 0
        # if type(overlay_list[0]) is list:
        #     overlay_iters = overlay_list[0]
        # else:
        #     overlay_iters = overlay_list
        # for i in overlay_iters:
        #     if len(i.shape) == 3:
        #         n_overlay_labels += 1
        #     elif len(i.shape) == 4:
        #         n_overlay_labels += i.shape[0]

        # return n_overlay_labels

    def view(self, plane='axial', three_planes=False, image_size=300, dynamic=True):
        # imopts = {'tools': ['hover'], 'width': 400, 'height': 400, 'cmap': 'gray'}
        imopts = {'tools': ['hover'], 'cmap': 'gray'}
        opts.defaults(
            opts.GridSpace(shared_xaxis=False, shared_yaxis=False,
                           fontsize={'title': 16, 'labels': 16, 'xticks': 12,
                                     'yticks': 12},
                           ),
            opts.Image(cmap='gray', tools=['hover'], xaxis=None,
                       yaxis=None, shared_axes=False),
            opts.Overlay(tools=['hover']),
            opts.NdOverlay(tools=['hover']),
        )

        self.set_size(image_size)
        if plane=='axial':
            a1, a2, a3 = 'x', 'y', 'z'
            invert=True
            pane_width = self.axial_width
            pane_height = self.axial_height
        elif plane=='coronal':
            a1, a2, a3 = 'x', 'z', 'y'
            pane_width = self.coronal_width
            pane_height = self.coronal_height
            invert=False
        elif plane=='sagittal':
            a1, a2, a3 = 'y', 'z', 'x'
            invert=False
            pane_width = self.sagittal_width
            pane_height = self.sagittal_height

        contrast_start_min = np.asscalar(self.ds.isel(subject_id=0,
                                                    label=0).image.quantile(0.01).values)
        contrast_start_max = np.asscalar(self.ds.isel(subject_id=0,
                                                    label=0).image.quantile(0.99).values)
        contrast_min = np.asscalar(self.ds.isel(subject_id=0, label=0).image.min().values)
        contrast_max = np.asscalar(self.ds.isel(subject_id=0, label=0).image.max().values)
        ctotal = contrast_max - contrast_min
        contrast_min -= ctotal*0.1
        contrast_max += ctotal*0.1

        cslider = pn.widgets.RangeSlider(start=contrast_min, end=contrast_max,
                                         value=(contrast_start_min, contrast_start_max),
                                         name='contrast')
        if 'overlay' in self.ds.data_vars:
            hv_ds_image = hv.Dataset(self.ds['image'])
            if self.verbose:
                print(hv_ds_image)
            hv_ds_overlay = hv.Dataset(self.ds['overlay'])
            if self.verbose:
                print(hv_ds_overlay)
            tooltips = [
                ('x', '@x'),
                ('y', '@y'),
                ('z', '@z'),
                ('image', '@image'),
                ('overlay', '@overlay')
            ]
            hover = HoverTool(tooltips=tooltips)
            if self.verbose:
                print('overlay_max_calc')
            first_subj_max = self.ds.isel(subject_id=0).overlay.max(dim=['x', 'y', 'z',
                                                                         'label']).compute()
            first_subj_min = self.ds.isel(subject_id=0).overlay.min(dim=['x', 'y', 'z',
                                                                         'label']).compute()
            if self.verbose:
                print('overlay_max_calc ready')
                print(first_subj_max)
            overlay_max = first_subj_max.max()
            alpha_slider = pn.widgets.FloatSlider(start=0, end=1, value=0.7, name='overlay transparency')
            cmap_select = pn.widgets.Select(name='Overlay Colormap', options=['Discrete',
                                                                              'Continuous'])

            if self.verbose:
                print('max thresh calc')
                print(first_subj_max.max())
            max_thresholds = first_subj_max.values
            if max_thresholds.size != 1:
                max_thresholds = sorted(set(max_thresholds))
            else:
                max_thresholds = [np.asscalar(max_thresholds)]
            # max_thresholds = sorted(list(set([first_subj.overlay.sel(overlay_label=i).values.max() for i in
            #                        first_subj.overlay_label])))
            if self.verbose:
                print('min thresh calc')
            min_thresholds = first_subj_min.values+1e-6
            if min_thresholds.size != 1:
                min_thresholds = sorted(set(min_thresholds))
            else:
                min_thresholds = [np.asscalar(min_thresholds)]
            # min_thresholds = sorted(list(set(first_subj_min.min())))
            # min_thresholds = sorted(list(set([first_subj.sel(overlay_label=i).min()+1e-6 for i in
            #                        first_subj.overlay_label])))
            ocslider = pn.widgets.DiscreteSlider(name='overlay max threshold', options=max_thresholds,
                                                value=max_thresholds[-1])
            if len(min_thresholds) == 1 and len(max_thresholds) == 1:
                thresh_toggle=0
                oclim = (min_thresholds[0], max_thresholds[0])

            elif len(min_thresholds) > 1 and len(max_thresholds) == 1:
                thresh_toggle=1
                ocslider_min = pn.widgets.DiscreteSlider(name='overlay min threshold',
                                                     options=min_thresholds,
                                                     value=min_thresholds[-1])
                @pn.depends(ocslider_min)
                def oclim(value):
                    return (value, max_thresholds[0])

            elif len(min_thresholds) == 1 and len(max_thresholds) > 1:
                thresh_toggle=2
                ocslider_max = pn.widgets.DiscreteSlider(name='overlay max threshold',
                                                     options=max_thresholds,
                                                     value=max_thresholds[-1])
                @pn.depends(ocslider_max)
                def oclim(value):
                    return (min_thresholds[0], value)

            else:
                thresh_toggle=3
                ocslider_min = pn.widgets.DiscreteSlider(name='overlay min threshold',
                                                     options=min_thresholds,
                                                     value=min_thresholds[-1])
                ocslider_max = pn.widgets.DiscreteSlider(name='overlay max threshold',
                                                     options=max_thresholds,
                                                     value=max_thresholds[-1])
                @pn.depends(ocslider_min, ocslider_max)
                def oclim(value_min, value_max):
                    return (value_min, value_max)

            if self.verbose:
                print(thresh_toggle)
            @pn.depends(cmap_select)
            def cmap_dict(value):
                d = {'Discrete': 'glasbey_hv','Continuous': 'viridis'}
                return d[value]


            # subj_viewer = SubjectViewer(ds=self.ds, subject_id_sel=list(self.ds.subject_id.values))

            if three_planes:
                squish_height = int(max(image_size*(len(self.ds.z)/len(self.ds.x)), image_size/2))
                # gridspace = hv.GridSpace(kdims=['plane', 'label'], label=f'{self.subject_id}')
                gridspace = hv.GridSpace(kdims=['plane', 'label'])
                for mod in self.ds.label.values:
                    gridspace['axial', mod] = hv_ds_image.select(label=mod).to(
                        hv.Image, ['x', 'y'], groupby=['subject_id', 'z'], vdims='image',
                        dynamic=dynamic).opts(
                            frame_width=self.axial_width, frame_height=self.axial_height).apply.opts(clim=cslider.param.value)
                    gridspace['coronal', mod] = hv_ds_image.select(label=mod).to(
                        hv.Image, ['x', 'z'], groupby=['subject_id', 'y'], vdims='image',
                        dynamic=dynamic).opts(frame_width=self.coronal_width,
                                              frame_height=self.coronal_height).apply.opts(clim=cslider.param.value)
                    gridspace['sagittal', mod] = hv_ds_image.select(label=mod).to(
                        hv.Image, ['y', 'z'], groupby=['subject_id', 'x'], vdims='image',
                        dynamic=dynamic).opts(frame_width=self.sagittal_width,
                                              frame_height=self.sagittal_height).apply.opts(clim=cslider.param.value)

                    gridspace['axial', mod] *= hv_ds_overlay.select(label=mod).to(
                        hv.Image, ['x', 'y'], groupby=['subject_id', 'z', 'overlay_label'], vdims='overlay',
                        dynamic=dynamic).opts(
                            cmap='glasbey_hv', clipping_colors={'min': 'transparent', 'NaN': 'transparent'},
                        ).redim.range(overlay=(0.1, overlay_max)).apply.opts(
                            alpha=alpha_slider.param.value, cmap=cmap_dict, clim=oclim)
                    gridspace['coronal', mod] *= hv_ds_overlay.select(label=mod).to(
                        hv.Image, ['x', 'z'], groupby=['subject_id', 'y', 'overlay_label'], vdims='overlay',
                        dynamic=dynamic).opts(
                            cmap='glasbey_hv', clipping_colors={'min': 'transparent', 'NaN': 'transparent'},
                        ).redim.range(overlay=(0.1, overlay_max)).apply.opts(
                            alpha=alpha_slider.param.value, cmap=cmap_dict, clim=oclim)
                    gridspace['sagittal', mod] *= hv_ds_overlay.select(label=mod).to(
                        hv.Image, ['y', 'z'], groupby=['subject_id', 'x', 'overlay_label'], vdims='overlay',
                        dynamic=dynamic).opts(
                            cmap='glasbey_hv', clipping_colors={'min': 'transparent', 'NaN': 'transparent'},
                        ).redim.range(overlay=(0.1, overlay_max)).apply.opts(
                            alpha=alpha_slider.param.value, cmap=cmap_dict, clim=oclim)

            else:
                squish_height = int(max(image_size*(len(self.ds.z)/len(self.ds.x)), image_size/2))
                # gridspace = hv.GridSpace(kdims=['label'], label=f'{self.subject_id}')
                if self.verbose:
                    print('init gridspace')
                # gridspace = hv.GridSpace(kdims=['label'])
                # for mod in self.ds.label:
                #     gridspace[mod] = hv_ds_image.select(label=mod).to(
                #         hv.Image, [a1, a2], groupby=[a3], vdims='image',
                #         dynamic=dynamic).opts(frame_width=image_size, frame_height=image_size,
                #                            ).apply.opts(clim=cslider.param.value)
                #     gridspace[mod] *= hv_ds_overlay.select(label=mod).to(
                #         hv.Image, [a1, a2], groupby=[a3, 'overlay_label'], vdims='overlay',
                #         dynamic=dynamic).opts(
                #             cmap='glasbey_hv', clipping_colors={'min': 'transparent'},
                #         ).redim.range(overlay=(1e-6, overlay_max)).apply.opts(
                #             alpha=alpha_slider.param.value, cmap=cmap_dict, clim=oclim)
                #     gridspace[mod] = gridspace[mod].opts(tools=['hover'])
                #     print(gridspace[mod])


                gridspace = hv_ds_image.to(
                    hv.Image, [a1, a2], vdims='image',
                    dynamic=dynamic).opts(frame_width=pane_width, frame_height=pane_height,
                                          ).apply.opts(clim=cslider.param.value)
                if self.verbose:
                    print(gridspace)
                gridspace *= hv_ds_overlay.to(
                    hv.Image, [a1, a2], vdims='overlay',
                    dynamic=dynamic).opts(
                        cmap='glasbey_hv', clipping_colors={'min': 'transparent', 'NaN': 'transparent'},
                    ).redim.range(overlay=(1e-6, overlay_max)).apply.opts(
                        alpha=alpha_slider.param.value, cmap=cmap_dict, clim=oclim)
                # print(gridspace)
                # print(gridspace)
                # gridspace = hv.DynamicMap(subj_viewer.load_subject).grid('label')
                gridspace = gridspace.grid('label')

        else:
            hv_ds = hv.Dataset(self.ds['image'])
            if three_planes:
                squish_height = int(max(image_size*(len(self.ds.z)/len(self.ds.x)), image_size/2))
                # gridspace = hv.GridSpace(kdims=['plane', 'label'], label=f'{self.subject_id}')
                gridspace = hv.GridSpace(kdims=['plane', 'label'])
                for mod in self.label:
                    gridspace['axial', mod] = hv_ds.select(label=mod).to(
                        hv.Image, ['x', 'y'], groupby=['subject_id', 'z'], vdims='image',
                        dynamic=dynamic).opts(frame_width=self.axial_width,
                                              frame_height=self.axial_height,
                                              invert_yaxis=False).apply.opts(clim=cslider.param.value)
                    gridspace['coronal', mod] = hv_ds.select(label=mod).to(
                        hv.Image, ['x', 'z'], groupby=['subject_id', 'y'], vdims='image',
                        dynamic=dynamic).opts(frame_width=self.coronal_width,
                                              frame_height=self.coronal_height).apply.opts(clim=cslider.param.value)
                    gridspace['sagittal', mod] = hv_ds.select(label=mod).to(
                        hv.Image, ['y', 'z'], groupby=['subject_id', 'x'], vdims='image',
                        dynamic=dynamic).opts(frame_width=self.sagittal_width,
                                              frame_height=self.sagittal_height).apply.opts(clim=cslider.param.value)

            else:
                squish_height = int(max(image_size*(len(self.ds.z)/len(self.ds.x)), image_size/2))
                # gridspace = hv.GridSpace(kdims=['label'], label=f'{self.subject_id}')
                gridspace = hv.GridSpace(kdims=['label'])
                for mod in self.label:
                    gridspace[mod] = hv_ds.select(label=mod).to(
                        hv.Image, [a1, a2], groupby=['subject_id', a3], vdims='image',
                        dynamic=dynamic).opts(frame_width=pane_width, frame_height=self.pane_height,
                                              shared_axes=False).apply.opts(clim=cslider.param.value)

        pn_layout = pn.pane.HoloViews(gridspace)
        wb = pn_layout.widget_box
        wb.append(cslider)
        if 'overlay' in self.ds.data_vars:
            wb.append(alpha_slider)
            wb.append(cmap_select)
            if thresh_toggle in [2, 3]:
                wb.append(ocslider_max)
            if thresh_toggle in [1, 3]:
                wb.append(ocslider_min)
        return pn.Row(wb, pn_layout)

    def set_size(self, size):
        init_spacing = self.ds.spacing.isel(subject_id=0, label=0).values
        x_spacing = init_spacing[0]
        y_spacing = init_spacing[1]
        z_spacing = init_spacing[2]

        self.axial_width = int(size)
        axial_scale = self.axial_width/(len(self.ds.x)*x_spacing)
        self.axial_height = int(y_spacing*len(self.ds.y)*axial_scale)

        self.coronal_width = int(size)
        coronal_scale = self.coronal_width/(len(self.ds.x)*x_spacing)
        self.coronal_height = int(z_spacing*len(self.ds.z)*coronal_scale)

        self.sagittal_width = int(size)
        sagittal_scale = self.sagittal_width/(len(self.ds.y)*y_spacing)
        self.sagittal_height = int(z_spacing*len(self.ds.z)*sagittal_scale)



class PyCohort(PyPatient):
    """Class for veiwing image and metadata for a cohort of patients."""

    def __init__(self, path, **kwargs):
        '''Initialize via reading the image and creating the xarray.'''
        self.ds = self.load_files(path)
        self.verbose = kwargs.get('verbose', False)


    def load_files(self, file_names):
        if type(file_names) is not list:
            file_names = str(file_names)
        else:
            file_names = [f for f in file_names if Path(f).exists()]

        if '*' in file_names or type(file_names) is list:
            # ds = xr.open_mfdataset(file_names, combine='nested', concat_dim='subject_id')
            ds = xr.open_mfdataset(file_names, concat_dim='subject_id', combine='nested').persist()
        else:
            ds = xr.open_dataset(file_names)
        return ds


# class SubjectViewer(param.Parameterized):
#     ds = param.ClassSelector(class_=xr.Dataset)
#     subject_id_sel = param.ObjectSelector(default='18725M')
#     slice_sel = param.Integer(default=0, bounds=(0,400))
#
#     @param.depends('subject_id_sel', 'ds', 'slice_sel')
#     def load_subject(self):
#         subj_ds = self.ds.sel(subject_id=self.subject_id_sel, z=[self.slice_sel],
#                               label='Expiratory CT').compute()
#         # return hv.Curve(df, ('date', 'Date'),
#         #                 self.variable).opts(framewise=True)
#
#         # gridspace = hv_ds_image.to(
#         #     hv.Image, [a1, a2], vdims='image',
#         #     dynamic=dynamic).opts(framewise=True, frame_width=image_size, frame_height=image_size,
#         #                           ).apply.opts(clim=cslider.param.value)
#         # hv_ds_image = hv.Dataset(subj_ds['image'])
#         # hv_ds_overlay = hv.Dataset(subj_ds['overlay'])
#         gridspace = hv.Image(subj_ds, kdims=['x', 'y'], vdims='image', dynamic=False).opts(framewise=True,
#                                                                                      frame_width=500,
#                                                                                      frame_height=500)
#
#         return gridspace
