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
 * @author: Maximilian F.S.J. Menger
 * @date: 18.04.2018
 * @version: 0.1.1
 *
 * modified by Marco Romanelli
 * @date: 28/07/2025
 * Added routine for reading in phases from QMout and so here it is modified accordingly
 *
 * Python Wrapper for the SHARC LIBRARY
 *
 * Main routine to setup the sharc driver.
 * uses interface.f90 to call sharc.
 */

#include <Python.h>
#include "structmember.h"
/*#ifdef __NUMPY__ALLOWED__*/
#include <numpy/arrayobject.h>
/*#endif*/
/* DEFINITIONS USED TO COMMUNICATE */
#include "data.inc"
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
// basic tools
#include "pysharc_tools.h"
#include "libsharc.h"
#include "python_2_3.h"

/*********************** GET INFO ********************************************/

/* get current atomid */
PyObject * get_atomid(void)
{

    PyObject * lst;

    int NAtoms = 0;
    get_natoms_(&NAtoms);

    int * IAn;
    IAn = (int *)malloc(NAtoms*sizeof(int));

    get_ian_(&NAtoms, IAn);

    lst = PyList_New(NAtoms);
    for (int i=0; i<NAtoms; i++){
        PyObject * pyfloat = PyInt_FromLong( *(IAn+i) );
        PyList_SetItem(lst, i, pyfloat);
    }
    free(IAn);

    return lst;
}

/* get current coordinates */
static char get_current_coordinates_docstring[] =
    "get_current_coordinates()\n\
    :return: lst";

static PyArrayObject * get_current_coordinates(PyObject * self, PyObject * args)
{

    int NAtoms = 0;
    get_natoms_(&NAtoms);

    const npy_intp dims[] = {NAtoms, 3};
    PyArrayObject * Crd  = ((PyArrayObject *)PyArray_ZEROS(2, dims, NPY_FLOAT64, 0));
    double * data = ((double *)PyArray_DATA(Crd));

    int ang;

    if (!PyArg_ParseTuple(args, "i", &ang))
        return NULL;

    get_current_coordinates_(&NAtoms, data, &ang);

    return Crd;
}
static char get_current_velocities_docstring[] =
    "get_current_velocities()\n\
    :return: lst";

static PyArrayObject * get_current_velocities(PyObject * self, PyObject * args)
{
    int Natoms = 0;
    get_natoms_(&Natoms);

    const npy_intp dims[] = {Natoms, 3};
    PyArrayObject * Crd = ((PyArrayObject *)PyArray_ZEROS(2, dims, NPY_FLOAT64, 0));
    double * data = ((double *)PyArray_DATA(Crd));
    get_current_velocities_(&Natoms, data);

    return Crd;
}


/* get_basic_info */
static char get_basic_info_docstring[] =
    "setup_sharc(fileName)\n\
    :return: int ";

static PyObject * get_basic_info(PyObject * self)
{
    int N_func_str = 4;
    char * info_names_str [] =
    { "states", "charge", "dt", "retain" } ;
    void (*get_info_str []) (char *) =
    { get_states_, get_charges_, get_dt_, get_retain_ };

    int N_func_int = 3;
    char * info_names_int [] =
    { "NAtoms", "NSteps", "istep" } ;
    void (*get_info_int []) (int *) =
    { get_natoms_, get_nsteps_, get_trajstep_ };


    int N_func_pyobj = 1;
    char * info_names_pyobj [] =
        {"IAn"//, "AtNames"
        };
    PyObject * (*get_info_pyobj []) (void) =
        { get_atomid//,   get_atom_names 
        };

    char * string;

    PyObject * dct;

    dct = PyDict_New();
//  Strings
    string = (char *)malloc(STRING_SIZE_S_*sizeof(char));
    for (int i=0; i < N_func_str; i++) {
        get_info_str[i](string);
        PyObject * pystring = PyString_FromString(string);
        if (pystring == NULL) {
            goto fail_string;
        }
        PyDict_SetItemString(dct, info_names_str[i], pystring);
    }
    free(string);
// Integers
    int ivalue = 0;
    for (int i=0; i < N_func_int; i++) {
        get_info_int[i](&ivalue);
        PyObject * pyint = PyInt_FromLong(ivalue);
        if (pyint == NULL) {
            goto fail_int;
        }
        PyDict_SetItemString(dct, info_names_int[i], pyint);
    }
// PyObject
    for (int i=0; i < N_func_pyobj; i++) {
        PyObject * pyobj = get_info_pyobj[i]();
        if (pyobj == NULL) {
            goto fail_int;
        }
        PyDict_SetItemString(dct, info_names_pyobj[i], pyobj);
    }
    

    return dct;
    fail_string:
        free(string);
        Py_XDECREF(dct);
        return NULL;
    fail_int:
        Py_XDECREF(dct);
        return NULL;
}

static int add_grad_compact(PyObject *dct, int icall, int nstates)
{
    if (nstates <= 0) {
        PyErr_SetString(PyExc_ValueError, "nstates must be > 0");
        return -1;
    }

    int8_t mode = 0;
    get_grad_mode_(&icall, &mode);

    PyObject *py_mode = PyLong_FromLong((long)mode);
    if (!py_mode)
        return -1;

    PyObject *py_mask = NULL;

    if (mode == 2) {  // SUBSET => allocate and fill mask
        size_t nwords = (size_t)((nstates + 63) / 64);
        uint64_t *words = (uint64_t*)malloc(nwords * sizeof(uint64_t));
        if (!words) {
            Py_DECREF(py_mode);
            PyErr_NoMemory();
            return -1;
        }

        fill_grad_mask_(&nstates, words);

        py_mask = PyBytes_FromStringAndSize(
            (const char*)words,
            (Py_ssize_t)(nwords * sizeof(uint64_t))
        );
        free(words);

        if (!py_mask) {
            Py_DECREF(py_mode);
            return -1;
        }
    } else {
        // NONE or ALL => empty mask
        py_mask = PyBytes_FromStringAndSize(NULL, 0);
        if (!py_mask) {
            Py_DECREF(py_mode);
            return -1;
        }
    }

    if (PyDict_SetItemString(dct, "grad_mode", py_mode) < 0 ||
        PyDict_SetItemString(dct, "grad_mask", py_mask) < 0) {
        Py_DECREF(py_mode);
        Py_DECREF(py_mask);
        return -1;
    }

    Py_DECREF(py_mode);
    Py_DECREF(py_mask);
    return 0;
}

static int add_nacdr_compact(PyObject *dct, int icall, int nstates)
{
    int8_t mode = 0;
    get_nacdr_mode_(&icall, &mode);

    PyObject *py_mode = PyLong_FromLong((long)mode);
    if (!py_mode)
        return -1;

    PyObject *py_mask = NULL;

    if (mode == 2)  // subset only
    {
        size_t nbits  = (size_t)nstates * (size_t)nstates;
        size_t nwords = (nbits + 63u) / 64u;

        uint64_t *words = malloc(nwords * sizeof(uint64_t));
        if (!words) {
            Py_DECREF(py_mode);
            PyErr_NoMemory();
            return -1;
        }

        fill_nacdr_mask_(&nstates, words);

        py_mask = PyBytes_FromStringAndSize(
            (const char*)words,
            (Py_ssize_t)(nwords * sizeof(uint64_t))
        );

        free(words);

        if (!py_mask) {
            Py_DECREF(py_mode);
            return -1;
        }
    }
    else
    {
        // EMPTY mask when mode != subset
        py_mask = PyBytes_FromStringAndSize(NULL, 0);
        if (!py_mask) {
            Py_DECREF(py_mode);
            return -1;
        }
    }

    if (PyDict_SetItemString(dct, "nacdr_mode", py_mode) < 0 ||
        PyDict_SetItemString(dct, "nacdr_mask", py_mask) < 0)
    {
        Py_DECREF(py_mode);
        Py_DECREF(py_mask);
        return -1;
    }

    Py_DECREF(py_mode);
    Py_DECREF(py_mask);
    return 0;
}


/* get_all_tasks */
static char get_all_tasks_docstring[] =
    "setup_sharc(fileName)\n\
    :return: int ";

static PyObject * get_all_tasks(PyObject * self, PyObject * args)
{
    int icall = 0;
    int nstates = 0;
    if (!PyArg_ParseTuple(args, "ii", &icall, &nstates))
        return NULL;

    PyObject *dct = PyDict_New();
    if (!dct) return NULL;

    int32_t step = 0;
    uint64_t mask = 0;
    get_tasks_mask_(&step, &icall, &mask);

    PyObject *py_step = PyLong_FromLong((long)step);
    PyObject *py_mask = PyLong_FromUnsignedLongLong((unsigned long long)mask);
    if (!py_step || !py_mask) {
        Py_XDECREF(py_step);
        Py_XDECREF(py_mask);
        Py_DECREF(dct);
        return NULL;
    }

    if (PyDict_SetItemString(dct, "step", py_step) < 0 ||
        PyDict_SetItemString(dct, "mask", py_mask) < 0) {
        Py_DECREF(py_step);
        Py_DECREF(py_mask);
        Py_DECREF(dct);
        return NULL;
    }
    Py_DECREF(py_step);
    Py_DECREF(py_mask);

    // grad
    if (add_grad_compact(dct, icall, nstates) < 0) {
        Py_DECREF(dct);
        return NULL;
    }

    // nacdr
    if (add_nacdr_compact(dct, icall, nstates) < 0) {
        Py_DECREF(dct);
        return NULL;
    }
    return dct;
}

/* include sharc main */
#include "pysharc_main.c"
/* include everything related to qmout */
#include "pysharc_QMout.c"
/* include everything related to qmin */
#include "pysharc_QMin.c"

// -----------------------------------------------------------------

/* set QMout */
static char set_qmout_docstring[] =
    "setup_sharc(qmout)\n\
    :qmout qmout: needs to be of type qmout ! \n\
    :return: None";

static PyObject * set_qmout(PyObject * self, PyObject * args)
{
    QMout * qmout;
    int icall = 0;
    if (!PyArg_ParseTuple(args, "Oi", &qmout, &icall))
        return NULL;

    if (!PyObject_TypeCheck(qmout, &QMoutType)){
        PyErr_SetString(PyExc_TypeError, "arg #1 needs to be of type QMout! ");
        return NULL;
    }

    const int iset_g = qmout->iset_g;
    const int iset_nacdr = qmout->iset_nacdr;

    // only properties that need to be changed, are done here
    postprocess_qmout_data_(&qmout->iset_h,
                              &qmout->iset_d,
                              &qmout->iset_g,
                              &qmout->iset_o,
                              &qmout->iset_phases,
                              &qmout->iset_nacdr
                            );
    /* set phases */
    //set_phases_(); Now proper phases reading routine is implemented and the call is in driver.py
    // Post process data after setting it
    int ISecond = 0;
    if (icall == 1) {
        post_process_data_(&ISecond);
    }
    /*    if nacdr/gradients were not in icall 1
     *    but iscond is true they need to be cleared!
     */
    if (ISecond == 1) {
        // clear memory!
        if (iset_g == 0) {
            clear_double(qmout->NStates * qmout->NAtoms * 3, qmout->gradient);
        }
        if (iset_nacdr == 0) {
            clear_double(qmout->NStates * qmout->NStates * qmout->NAtoms * 3, 
                    qmout->nacdr);
        }
    }
    // Return ISecond
    return Py_BuildValue("i", ISecond);
}

// -----------------------------------------------------------------

/* SHARC METHODS */
static PyMethodDef SHARC_METHODS[] = {
    /* QMout */
    {"set_qmout", (PyCFunction)set_qmout, METH_VARARGS, set_qmout_docstring},
    /* GET INFO  */
    {"get_basic_info", (PyCFunction)get_basic_info, METH_NOARGS, get_basic_info_docstring},
    {"get_all_tasks", (PyCFunction)get_all_tasks, METH_VARARGS, get_all_tasks_docstring},
    {"get_crd", (PyCFunction)get_current_coordinates, METH_VARARGS, get_current_coordinates_docstring},
    {"get_vel", (PyCFunction)get_current_velocities, METH_VARARGS, get_current_velocities_docstring},
    /* sharc initial qm */
    {"initial_qm_pre", (PyCFunction)initial_qm_pre, METH_NOARGS, initial_qm_pre_docstring},
    {"initial_qm_post", (PyCFunction)initial_qm_post, METH_NOARGS, initial_qm_post_docstring},
    /* SHARC MAIN ROUTINES*/
    {"setup_sharc", (PyCFunction)setup_sharc, METH_VARARGS, setup_sharc_docstring},
    {"initial_step", (PyCFunction)initial_step, METH_VARARGS, initial_step_docstring},
    {"verlet_xstep", (PyCFunction)verlet_xstep, METH_VARARGS, verlet_xstep_docstring},
    {"verlet_vstep", (PyCFunction)verlet_vstep, METH_VARARGS, verlet_vstep_docstring},
    {"verlet_finalize", (PyCFunction)verlet_finalize, METH_VARARGS, verlet_finalize_docstring},
    {"finalize_sharc", (PyCFunction)finalize_sharc, METH_NOARGS, finalize_sharc_docstring},
    {"error_finalize_sharc", (PyCFunction)error_finalize_sharc, METH_VARARGS, error_finalize_sharc_docstring},
    /* SENTINEL */
    {NULL, NULL, 0, NULL}
};

// define sharc_module_init
static PyObject *
sharc_module_init(void)
{
    // check if Python Type is ready!
    if (PyType_Ready(&QMoutType) < 0 )
        return NULL;
    // check if Python Type is ready!
    if (PyType_Ready(&QMinType) < 0 )
        return NULL;
    // Define Module
    PyObject * mod;
    MOD_DEF(mod, sharc, "sharc",
      "Python API for the SHARC MD code",
      SHARC_METHODS,
      NULL, NULL, NULL, NULL)

    if (mod == NULL)
        return NULL;
    // set  QMout module
    Py_INCREF(&QMoutType);
    PyModule_AddObject(mod, "QMout",
            (PyObject *)&QMoutType);
    // set  QMin module
    Py_INCREF(&QMinType);
    PyModule_AddObject(mod, "QMin",
            (PyObject *)&QMinType);
    /* Load `numpy` */
    import_array();
    return mod;
}


/* DEFINE NEW MODULE SHARC */
MOD_INIT(sharc)
{
#if PY_MAJOR_VERSION >= 3
    return sharc_module_init();
#else
    PyObject * mod=sharc_module_init();
    if (mod == NULL)
        return ;
#endif
}
