
# Modified based on the repository https://github.com/otaheri/GRAB (Author: Omid Taheri)


import numpy as np

import torch
import torch.nn as nn
from smplx.lbs import batch_rodrigues
from collections import namedtuple

model_output = namedtuple('output', ['vertices', 'global_orient', 'transl', 'center', 'vn'])

class ObjectModel(nn.Module):

    def __init__(self,
                 v_template,
                 batch_size=1,
                 vn=None,
                 center=None,
                 dtype=torch.float32):
        ''' 3D rigid object model

                Parameters
                ----------
                v_template: np.array Vx3, dtype = np.float32
                    The vertices of the object
                batch_size: int, N, optional
                    The batch size used for creating the model variables

                dtype: torch.dtype
                    The data type for the created variables
            '''

        super(ObjectModel, self).__init__()


        self.dtype = dtype

        # Mean template vertices
        v_template = np.repeat(v_template[np.newaxis], batch_size, axis=0)
        self.register_buffer('v_template', torch.tensor(v_template, dtype=dtype))

        v_center = np.repeat(center[np.newaxis], batch_size, axis=0)
        self.register_buffer('v_center', torch.tensor(v_center, dtype=dtype))

        transl = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))

        global_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('global_orient', nn.Parameter(global_orient, requires_grad=True))

        self.batch_size = batch_size

        self.obj_normal = False
        if vn is not None:
            self.obj_normal = True
            vn = np.repeat(vn[np.newaxis], batch_size, axis=0)
            self.register_buffer('vn', torch.tensor(vn, dtype=dtype))


    def forward(self, global_orient=None, transl=None, v_template=None, vn=None, **kwargs):

        ''' Forward pass for the object model

        Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)

            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            v_template: torch.tensor, optional, shape BxVx3
                The new object vertices to overwrite the default vertices

        Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''
        

        if global_orient is None:
            global_orient = self.global_orient
        if transl is None:
            transl = self.transl
        if v_template is None:
            v_template = self.v_template
        if vn is None and self.obj_normal:
            vn = self.vn

        rot_mats = batch_rodrigues(global_orient.view(-1, 3)).view([self.batch_size, 3, 3])

        vertices = torch.matmul(v_template, rot_mats) + transl.unsqueeze(dim=1)
        center = torch.matmul(self.v_center, rot_mats) + transl.unsqueeze(dim=1)

        if self.obj_normal:
            vn = torch.matmul(vn, rot_mats)

        output = model_output(vertices=vertices,
                              global_orient=global_orient,
                              transl=transl,
                              center=center,
                              vn=vn if self.obj_normal else None)

        return output

