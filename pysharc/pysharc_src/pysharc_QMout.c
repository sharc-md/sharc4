//******************************************
//
//    SHARC Program Suite
//
//    Copyright (c) 2025 University of Vienna
//
//    This file is part of SHARC.
//
//    SHARC is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    SHARC is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************


/*
 * @author: Maximilian F.S.J. Menger, Severin Polonius
 * @date: 29.07.2024
 * @version: 0.1.1 
 *
 * Python Wrapper for the SHARC LIBRARY
 * 
 * DEFINES the QMout object of the SHARC LIBRARY
 */

typedef struct {
    PyObject_HEAD
    PyObject * interface_name; 
    int NAtoms;
    int NStates;
    double * gradient;
    double * nacdr;
    double complex * hamiltonian;
    double complex * dipole_mom;
    double complex * overlap;
    int iset_h;
    int iset_g;
    int iset_d;
    int iset_o;
    int iset_nacdr;
    int imem;
} QMout;

static void
QMout_dealloc(QMout * self)
{
    Py_XDECREF(self->interface_name);
#ifdef __OWN_SPACE_QMout__
    free(self->hamiltonian);
    free(self->gradient);
    free(self->dipole_mom);
    free(self->overlap);
    free(self->nacdr);
#endif
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
QMout_new(PyTypeObject * type, PyObject *args, PyObject *kwds)
{
    QMout * self;

    self = (QMout *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->interface_name = PyString_FromString("");
        if (self->interface_name == NULL) {
            Py_DECREF(self);
            return NULL;
        }

        self->NAtoms = 0;
        self->NStates = 0;
        self->iset_h = 0;
        self->iset_g = 0;
        self->iset_d = 0;
        self->iset_o = 0;
        self->iset_nacdr = 0;
#ifdef __OWN_SPACE_QMout__
        self->imem = 0;
#else 
        self->imem = 1;
#endif
    }

    return (PyObject *)self;
}

static int
QMout_init(QMout *self, PyObject *args, PyObject *kwds)
{
    PyObject * interface_name=NULL;
    PyObject * tmp=NULL;

    static char *kwlist[] = {"interface", "NAtoms", "NStates", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oii", kwlist,
                                      &interface_name, &self->NAtoms,
                                      &self->NStates))
        goto fail;

    if (interface_name) {
        tmp = self->interface_name;
        Py_INCREF(interface_name);
        self->interface_name = interface_name;
        Py_XDECREF(tmp);
    }

#ifdef __OWN_SPACE_QMout__
    /* Allocate memory for the properties ! */
    self->gradient= (double *)malloc(self->NStates
             * self->NAtoms * 3 * sizeof(double));
    /* these are huge data chuncks, keep that in mind! */
    self->nacdr = (double *)malloc(self->NStates * self->NStates
             * self->NAtoms * 3 * sizeof(double));
    self->hamiltonian = (double complex *) malloc(self->NStates
             * self->NStates * sizeof(double complex));
    self->dipole_mom= (double complex *) malloc(3 * self->NStates
             * self->NStates * sizeof(double complex));
    self->overlap = (double complex *) malloc(self->NStates
             * self->NStates * sizeof(double complex));
    /* if fail goto fail */
    if ( (self->hamiltonian == NULL) ||
         (self->gradient== NULL)     ||
         (self->dipole_mom == NULL)  ||
         (self->overlap == NULL) ) {
            goto fail;
    }
#else
    double complex ** H_ptr = &self->hamiltonian;
    double complex ** DM_ptr = &self->dipole_mom;
    double complex ** Ov_ptr = &self->overlap;
    double ** G_ptr = &self->gradient;
    double ** NACDR_ptr = &self->nacdr;
    setPointers( (double complex **)H_ptr, 
                 (double complex **)DM_ptr, 
                 (double complex **)Ov_ptr, 
                 (double **)G_ptr, 
                 (double **)NACDR_ptr);
#endif

    set_phases_();
    return 0;

  fail:
    //self->tp_dealloc(type, 0);
    return -1;
}


static PyMemberDef QMout_members[] = {
    // do not expose anything else to the python api
    {"interface", T_OBJECT_EX, offsetof(QMout, interface_name), 0, "interface name"},
    {NULL}  /* Sentinel */
};

static PyObject *
QMout_printInfo(QMout * self)
{
#if PY_MAJOR_VERSION < 3 
    printf("QMout file for Interface: '%s'\n", PyString_AsString(self->interface_name));
    printf("NAtoms = %d\nNStates = %d\n", self->NAtoms, self->NStates);
#endif

    for (int istate=0; istate < self->NStates; istate++){
        printf("Gradient of state '%d'\n", istate);
        for (int iatom=0; iatom <  self->NAtoms; iatom++){
            printf("Atoms '%d' : ", iatom);
            for( int ixyz=0; ixyz < 3; ixyz++){
                printf("%lf  ", *(self->gradient + istate*(self->NAtoms*3) + iatom*3 + ixyz));
            }
            printf("\n");
        }
    }


    Py_RETURN_NONE;
}

static PyObject *
QMout_printAll(QMout * self)
{
#if PY_MAJOR_VERSION < 3 
    printf("QMout file for Interface: '%s'\n", PyString_AsString(self->interface_name));
    printf("NAtoms = %d\nNStates = %d\n", self->NAtoms, self->NStates);
#endif

    if (self->iset_h == 1) {
        printf("HAMILTONIAN\n");
        for (int istate=0; istate < self->NStates; istate++){
            for (int jstate=0; jstate <  self->NStates; jstate++){
                    double complex value = *(self->hamiltonian + istate*(self->NStates) + jstate);
                    printf("%lf + %lf * i    ", creal(value), cimag(value));
            }
            printf("\n");
        }
    }
    if (self->iset_g == 1){
        printf("Gradients\n");
        for (int istate=0; istate < self->NStates; istate++){
            printf("Gradient of state '%d'\n", istate);
            for (int iatom=0; iatom <  self->NAtoms; iatom++){
                printf("Atoms '%d' : ", iatom);
                for( int ixyz=0; ixyz < 3; ixyz++){
                    printf("%lf  ", *(self->gradient + istate*(self->NAtoms*3) + iatom*3 + ixyz));
                }
                printf("\n");
            }
        }
    }

    if (self->iset_d == 1) {
        printf("DM\n");
        for (int k=0; k < 3; k++){
            printf("DM xyz = '%d'", k);
            for (int istate=0; istate < self->NStates; istate++){
                for (int jstate=0; jstate <  self->NStates; jstate++){
                        double complex value = *(self->dipole_mom + istate*(self->NStates) + jstate);
                        printf("%lf + %lf * i    ", creal(value), cimag(value));
                }
                printf("\n");
            }
        }
    }

    if (self->iset_o == 1) {
        printf("OVERLAP\n");
        for (int istate=0; istate < self->NStates; istate++){
            for (int jstate=0; jstate <  self->NStates; jstate++){
                    double complex value = *(self->overlap + istate*(self->NStates) + jstate);
                    printf("%lf + %lf * i    ", creal(value), cimag(value));
            }
            printf("\n");
        }
    }

    Py_RETURN_NONE;
}

static PyObject *
QMout_set_gradient(QMout * self, PyObject * args)
{
    PyObject * gradient;
    PyObject * key;
    PyObject * value;
    int icall;
    long pos = 0;
    double * state_gradient;

    double scale = 0.0;
    double soc_scale = 0.0;

    get_scalingfactor_(&scale, &soc_scale);

    if (!PyArg_ParseTuple(args, "Oi", &gradient, &icall))
        return NULL;

    /* need to be python dict */
    if ( !PyDict_Check(gradient)) 
        goto fail;
    /* Clear the Gradient only in first run! */
    if (icall == 1) {
        clear_double(self->NStates*self->NAtoms*3, self->gradient);
    }

    /* allocate memory for the state_gradient */
    state_gradient = (double *) malloc(((self->NAtoms)*3)*sizeof(double));
    /* loop over all elments in the list */
    while (PyDict_Next(gradient, &pos, &key, &value)) {
        int IState = PyInt_AsLong(key);
        if (!PyList_Check(value) ||  (PyList_Size(value) != self->NAtoms) )
            goto fail;
        /* assume that else the size etc. is set correctly */
        for (int iatom=0; iatom < self->NAtoms; iatom++){
            PyObject * atom_grad = PyList_GetItem(value, iatom);
            for (int j=0; j<3; j++){
                *(state_gradient + iatom*3 + j) = PyFloat_AsDouble(PyList_GetItem(atom_grad, j));
            }
        }
        /* set state gradient */
#ifdef __OWN_SPACE_QMout__
        set_gradient(self->gradient, self->NAtoms, IState, state_gradient, scale);
#else
        set_gradient_in_sharc_order(self->gradient, 
                self->NAtoms, self->NStates, IState, state_gradient, scale);
#endif
    }
    /* free memory */
    free(state_gradient);
    self->iset_g = 1;
    Py_RETURN_NONE;
    fail:
        Py_XDECREF(gradient);
        Py_XDECREF(key);
        Py_XDECREF(value);
        free(state_gradient);
        return NULL;
}


static PyObject *
QMout_set_gradient_full_array(QMout * self, PyObject * args)
{
   PyArrayObject * grad=NULL;

   if (!PyArg_ParseTuple(args, "O", &grad)){
       return NULL;
   }

   if (grad == NULL) {
      return NULL;
   }


   if (!PyArray_Check(grad)) {
     PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array");
     return NULL;
   }
   

   npy_intp* dim= PyArray_DIMS(grad);
   int num_dims = PyArray_NDIM(grad);
   if (num_dims != 3) {
     PyErr_SetString(PyExc_TypeError, "Input array must be 3d!");
     goto fail;
   }
   if (dim[0] != self->NStates || dim[1] != self->NAtoms || dim[2] != 3) {
     PyErr_SetString(PyExc_TypeError, "Input array must be of dim Nstates x Natoms x 3 !");
     goto fail;
   }

   /* Clear the gradient vector! */
   /*Not needed if every element is assigned*/
   /*clear_double(self->NStates*self->NAtoms*3, self->gradient);*/

   /* set state gradient  */
   for (size_t istate=0; istate<self->NStates; istate++){
        for (size_t a=0; a<self->NAtoms; a++){
            for (size_t x=0; x<3; x++){
              // SHARC order: 3*natoms*nstates)
              *(self->gradient +  x * self->NAtoms * self->NStates + a * self->NStates + istate) = *((double *)PyArray_GETPTR3(grad, istate, a, x));
            }
        }
   }

   /* free memory */
   self->iset_g = 1;
   /*Py_DECREF(grad);*/
   Py_RETURN_NONE;
   fail:
       Py_XDECREF(grad);
       return NULL;
}

static PyObject *
QMout_set_hamiltonian(QMout * self, PyObject * args)
{
   PyArrayObject * hamiltonian=NULL;

   if (!PyArg_ParseTuple(args, "O", &hamiltonian)){
        return NULL;
   }

   if (hamiltonian == NULL) {
      return NULL;
   }

    /* needs to be array */
   if (!PyArray_Check(hamiltonian)) {
     PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array");
     return NULL;
   }

   npy_intp* dim= PyArray_DIMS(hamiltonian);
   int num_dims = PyArray_NDIM(hamiltonian);
   if (num_dims != 2) {
     PyErr_SetString(PyExc_TypeError, "Input array must be 2d!");
     goto fail;
   }
   if (dim[0] != self->NStates || dim[1] != self->NStates) {
     PyErr_SetString(PyExc_TypeError, "Input array must be of dim 3 x Nstates x Nstates!");
     goto fail;
   }


   int is_complex = PyArray_TYPE(hamiltonian) == NPY_COMPLEX128;
    /* Clear the overlap matrix! */
   /*Not needed if every element is assigned*/
    /*clear_complex_double(self->NStates*self->NStates, self->hamiltonian);*/


   for (int is=0; is < self->NStates; is++){
       for (int js =0; js < self->NStates; js++){
         if (is_complex) {
           *(self->hamiltonian + (js*self->NStates) + is) = *((double complex *)PyArray_GETPTR2(hamiltonian, is, js));
         }
         else {
           *(self->hamiltonian + (js*self->NStates) + is) = *((double *)PyArray_GETPTR2(hamiltonian, is, js)) + 0. * _Complex_I;
         }
       }
   }
   self->iset_h = 1;
   Py_RETURN_NONE;
   fail:
       Py_XDECREF(hamiltonian);
       return NULL;
}

static PyObject *
QMout_set_dipolemoment(QMout * self, PyObject * args)
{
   PyArrayObject * dip=NULL;
   /*double complex complex_value;*/
   /*PyFloat * pyfloat;*/

   if (!PyArg_ParseTuple(args, "O", &dip)){
        return NULL;
   }

   if (dip == NULL) {
      return NULL;
   }

    /* needs to be array */
   if (!PyArray_Check(dip)) {
     PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array");
     return NULL;
   }

   npy_intp* dim= PyArray_DIMS(dip);
   int num_dims = PyArray_NDIM(dip);
   if (num_dims != 3) {
     PyErr_SetString(PyExc_TypeError, "Input array must be 3d!");
     goto fail;
   }
   if (dim[0] != 3 || dim[1] != self->NStates || dim[2] != self->NStates) {
     PyErr_SetString(PyExc_TypeError, "Input array must be of dim 3 x Nstates x Nstates!");
     goto fail;
   }

   int is_complex = PyArray_TYPE(dip) == NPY_COMPLEX128;

    /* Clear the overlap matrix! */
   /*Not needed if every element is assigned*/
    /*clear_complex_double(3*self->NStates*self->NStates, self->dipole_mom);*/


   for (int k = 0; k < 3; k++){
       for (int is=0; is < self->NStates; is++){
           for (int js =0; js < self->NStates; js++){
               if (is_complex) {
                  *(self->dipole_mom + (k * self->NStates * self->NStates) + (js*self->NStates) + is) = *((double complex *)PyArray_GETPTR3(dip, k, is, js));
               }
               else {
                  *(self->dipole_mom + (k * self->NStates * self->NStates) + (js*self->NStates) + is) = *((double *)PyArray_GETPTR3(dip, k, is, js)) + 0. * _Complex_I;
               }
           }
      }
   }
   self->iset_d = 1;
   Py_RETURN_NONE;
   fail:
       Py_XDECREF(dip);
       return NULL;
}

static PyObject *
QMout_set_overlap(QMout * self, PyObject * args)
{
   PyArrayObject * overlap=NULL;
   /*double complex complex_value;*/
   /*PyFloat * value;*/

   if (!PyArg_ParseTuple(args, "O", &overlap)){
        return NULL;
   }

   if (overlap == NULL) {
      return NULL;
   }

    /* needs to be array */
   if (!PyArray_Check(overlap)) {
     PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array");
     return NULL;
   }

   npy_intp* dim= PyArray_DIMS(overlap);
   int num_dims = PyArray_NDIM(overlap);
   if (num_dims != 2) {
     PyErr_SetString(PyExc_TypeError, "Input array must be 2d!");
     goto fail;
   }
   if (dim[0] != self->NStates || dim[1] != self->NStates) {
     PyErr_SetString(PyExc_TypeError, "Input array must be of dim Nstates x Nstates!");
     goto fail;
   }
   int is_complex = PyArray_TYPE(overlap) == NPY_COMPLEX128;


   /* Clear the overlap matrix! */
   /*Not needed if every element is assigned*/
   /*clear_complex_double(self->NStates*self->NStates, self->overlap);*/

   for (int is=0; is < self->NStates; is++){
       for (int js =0; js < self->NStates; js++){

         if (is_complex) {
           *(self->overlap + self->NStates*js + is) = *((double complex *) PyArray_GETPTR2(overlap, is,js));
         }
         else {
           *(self->overlap + self->NStates*js + is) = *((double *) PyArray_GETPTR2(overlap, is,js)) + 0. * _Complex_I;
         }
       }
   }

   self->iset_o = 1;
   Py_RETURN_NONE;
   fail:
      Py_XDECREF(overlap);
      return NULL;
}

static PyObject *
QMout_set_nacdr(QMout * self, PyObject * args)
{
    PyObject * nacdr;
    PyObject * key_1;
    PyObject * value_1;
    PyObject * key_2;
    PyObject * value_2;
    int icall = 0;
    long ipos_1 = 0;
    double * state_state_nac;

    if (!PyArg_ParseTuple(args, "Oi", &nacdr, &icall))
        return NULL;

    /* need to be python dict */
    if ( !PyDict_Check(nacdr)) 
        goto fail;
    /* Clear the NAC vector! */
    if (icall == 1) {
        clear_double(self->NStates*self->NStates*self->NAtoms*3, self->nacdr);
    }
    /* allocate memory for the state_state_nac*/
    state_state_nac = (double *) malloc(((self->NAtoms)*3)*sizeof(double));
    /* loop over all elments in the list */
    while (PyDict_Next(nacdr, &ipos_1, &key_1, &value_1)) {
        int IState = PyInt_AsLong(key_1);
        /* if you screw it up, it is not my fault! */
        if (!PyDict_Check(value_1) ) 
            goto fail;
        long ipos_2 = 0;
        while (PyDict_Next(value_1, &ipos_2, &key_2, &value_2)) {
            if (!PyList_Check(value_2) ||  (PyList_Size(value_2) != self->NAtoms) )
                goto fail;
            
            int JState = PyInt_AsLong(key_2);
            /* state_state_nac = list * NAtoms of list of 3 floats !  */
            for (int iatom=0; iatom < self->NAtoms; iatom++){
                PyObject * atom_nac = PyList_GetItem(value_2, iatom);
                for (int j=0; j<3; j++){
                    *(state_state_nac + iatom*3 + j) = PyFloat_AsDouble(PyList_GetItem(atom_nac, j));
                }
            }
        /* set state gradient  */
#ifdef __OWN_SPACE_QMout__
            set_nacdr(self->nacdr, self->NAtoms, self->NStates, IState, JState, state_state_nac);
#else
            set_nacdr_in_sharc_order(self->nacdr, 
                self->NAtoms, self->NStates, IState, JState, state_state_nac);
#endif

        }
    }
    /* free memory */
    free(state_state_nac);
    self->iset_nacdr = 1;
    Py_RETURN_NONE;
    fail:
        Py_XDECREF(nacdr);
        Py_XDECREF(key_1);
        Py_XDECREF(value_1);
        Py_XDECREF(key_2);
        Py_XDECREF(value_2);
        free(state_state_nac);
        return NULL;
}

static PyObject *
QMout_set_nacdr_full_array(QMout * self, PyObject * args)
{
   PyArrayObject * nacdr=NULL;

   if (!PyArg_ParseTuple(args, "O", &nacdr)){
       return NULL;
   }

   if (nacdr == NULL) {
      return NULL;
   }


   if (!PyArray_Check(nacdr)) {
     PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array");
     return NULL;
   }
   
   /*nacdr = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);*/



   npy_intp* dim= PyArray_DIMS(nacdr);
   int num_dims = PyArray_NDIM(nacdr);
   if (num_dims != 4) {
     PyErr_SetString(PyExc_TypeError, "Input array must be 4d!");
     goto fail;
   }
   if (dim[0] != self->NStates || dim[1] != self->NStates || dim[2] != self->NAtoms || dim[3] != 3) {
     PyErr_SetString(PyExc_TypeError, "Input array must be of dim Nstates x Nstates x Natoms x 3 !");
     goto fail;
   }

   /* Clear the NAC vector! */
   /*Not needed if every element is assigned*/
   /*clear_double(self->NStates*self->NStates*self->NAtoms*3, self->nacdr);*/

   for (size_t istate=0; istate<self->NStates; istate++){
       for (size_t jstate=0; jstate<self->NStates; jstate++){
           for (size_t a=0; a<self->NAtoms; a++){
               for (size_t x=0; x<3; x++){
                 // SHARC order: 3*natoms*nstates*nstates)
                 *(self->nacdr +  x * self->NAtoms * self->NStates * self->NStates + a * self->NStates * self->NStates + jstate * self->NStates + istate) = *((double *)PyArray_GETPTR4(nacdr, istate, jstate, a, x));
                 // idx++;
               }
           }

       }
   }

   /* free memory */
   self->iset_nacdr = 1;
   /*Py_DECREF(nacdr);*/
   Py_RETURN_NONE;
   fail:
       Py_XDECREF(nacdr);
       return NULL;
}

static PyMethodDef QMout_methods[] = {
    {"set_hamiltonian", (PyCFunction)QMout_set_hamiltonian, METH_VARARGS,
     "enters a list of list of [nstate][nstate], type: complex or float "},
    {"set_gradient", (PyCFunction)QMout_set_gradient, METH_VARARGS,
     "enters a dict of lists, grad[IState] = [NAtoms][3], type: floats" },
    {"set_gradient_full_array", (PyCFunction)QMout_set_gradient_full_array, METH_VARARGS,
     "gradient ndarray [nstates][NAtoms][3], dtype: float  "},
    {"set_dipolemoment", (PyCFunction)QMout_set_dipolemoment, METH_VARARGS,
     "enters a list of list of list of [3][nstate][nstate], type: complex or float"},
    {"set_overlap", (PyCFunction)QMout_set_overlap, METH_VARARGS,
     "enters a list of list of [nstate][nstate], type: complex or float "},
    {"set_nacdr", (PyCFunction)QMout_set_nacdr, METH_VARARGS,
     "enters dict of dics nacs[istate][jstate] = [NAtoms][3], type: float  "},
    {"set_nacdr_full_array", (PyCFunction)QMout_set_nacdr_full_array, METH_VARARGS,
     "nacdr ndarray [nstates][nstates][NAtoms][3], dtype: float  "},
    {"printInfos", (PyCFunction)QMout_printInfo, METH_NOARGS,
        "print info about system" },
    {"printAll", (PyCFunction)QMout_printAll, METH_NOARGS,
        "print info about system" },
    {NULL}  /* Sentinel */
};


static PyTypeObject QMoutType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "sharc.QMout",             /* tp_name */
    sizeof(QMout),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)QMout_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "QMout object for sharc",  /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    QMout_methods,             /* tp_methods */
    QMout_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)QMout_init,      /* tp_init */
    0,                         /* tp_alloc */
    QMout_new,                 /* tp_new */
};

