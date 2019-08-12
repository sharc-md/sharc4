/*
 * @author: Maximilian F.S.J. Menger
 * @date: 18.04.2018
 * @version: 0.1.1 
 *
 * Python Wrapper for the SHARC LIBRARY
 * 
 * DEFINES the functions responsible for the
 * main sharc driver!
 *
 */

/*********************** INITIAL QM ******************************************/

/* initial pre qm */
static char  initial_qm_pre_docstring[] =
    "setup_sharc(fileName)\n\
    :return: int ";

static PyObject * 
initial_qm_pre(PyObject * self)
{
        initial_qm_pre_();
        Py_RETURN_NONE;
}

/* initial post qm */
static char  initial_qm_post_docstring[] =
    "setup_sharc(fileName)\n\
    :return: int ";

static PyObject * 
initial_qm_post(PyObject * self)
{
        initial_qm_post_();
        Py_RETURN_NONE;
}


/*********************** SHARC MAIN ROUTINES *********************************/
/* SETUP SHARC */
static char setup_sharc_docstring[] =
    "setup_sharc(fileName)\n\
    :return: int ";

static PyObject * 
setup_sharc(PyObject * self, PyObject * args)
{
        char * input_string;
        int IRestart = 0;
        if (!PyArg_ParseTuple(args, "s", &input_string))
            return NULL;
        setup_sharc_(input_string, &IRestart);
        return Py_BuildValue("i", IRestart);
}

/* INITIAL STEP */
static char initial_step_docstring[] =
    "setup_sharc(fileName)\n\
    :return: int ";

static PyObject * 
initial_step(PyObject * self, PyObject * args)
{
        int IRestart = 0;
        if (!PyArg_ParseTuple(args, "i", &IRestart))
            return NULL;
        //do_initial_step_2_(&IRestart);
        initial_step_(&IRestart);
        Py_RETURN_NONE;
}

/* Verlet XSTEP */
static char verlet_xstep_docstring[] =
    "setup_sharc(fileName)\n\
    :return: int ";

static PyObject * 
verlet_xstep(PyObject * self, PyObject * args)
{
        int I_Step = 0;
        if (!PyArg_ParseTuple(args, "i", &I_Step))
            return NULL;
        verlet_xstep_(&I_Step);
        Py_RETURN_NONE;
}
/* Verlet VSTEP */
static char verlet_vstep_docstring[] =
    "setup_sharc(fileName)\n\
    :return: int ";
static PyObject * verlet_vstep(PyObject * self)
{
    int iredo = 0;
    verlet_vstep_(&iredo);
    return Py_BuildValue("i", iredo);
}
/* Verlet Finalize */
static char verlet_finalize_docstring[] =
    "setup_sharc(fileName)\n\
    :return: int ";

static PyObject * verlet_finalize(PyObject* self, PyObject* args)
{
        int IExit = 0;
        int iskip = 0;
        if (!PyArg_ParseTuple(args, "i", &iskip))
            return NULL;
        verlet_finalize_(&IExit, &iskip);
        return Py_BuildValue("i", IExit);
}

/* FINALIZE SHARC */
static char finalize_sharc_docstring[] =
    "finalize_sharc()\n\
    :return: None";

static PyObject * finalize_sharc(PyObject * self)
{
        finalize_sharc_();
        Py_RETURN_NONE;
}

/* ERROR FINALIZE SHARC */
static char error_finalize_sharc_docstring[] = "error_finalize_sharc()\n:return: None";

static PyObject * error_finalize_sharc(PyObject * self)
{
        error_finalize_sharc_();
        Py_RETURN_NONE;
}

