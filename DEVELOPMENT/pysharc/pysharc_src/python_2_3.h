/*
 * @author: Maximilian F.S.J. Menger
 * @date: 18.04.2018
 * @version: 0.1.1
 *
 * Python Wrapper for the SHARC LIBRARY
 *
 * Preprocessor flags to ensure compatible with 
 * python 2 and python 3
 *
 */
// MOD_INIT
#if PY_MAJOR_VERSION >= 3
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#else
    #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
#endif
// MOD_DEF
#if PY_MAJOR_VERSION >= 3
    #define MOD_DEF(ob,modname, name, doc, methods, reload, traverse, clear, free) \
        static struct PyModuleDef MODULE_DEF_##modname = {                    \
                    PyModuleDef_HEAD_INIT,                                 \
                    name,                                                \
                    doc,                                                   \
                    -1,                                                    \
                    methods,                                               \
                    reload,                                                \
                    traverse,                                              \
                    clear,                                                 \
                    free,                                                  \
        };                                                                 \
        ob = PyModule_Create(&MODULE_DEF_##modname);
#else
    #define MOD_DEF(ob, modname, name, doc, methods, reload, traverse, clear, free) \
        ob = Py_InitModule3(name, methods, doc);
#endif
// INIT_ERROR
#if PY_MAJOR_VERSION >= 3
    #define INIT_ERROR return NULL
#else
    #define INIT_ERROR return
#endif
// others
#if PY_MAJOR_VERSION >= 3
    #define PyInt_FromLong PyLong_FromLong
    #define PyInt_AsLong PyLong_AsLong
// Strings...
    #define PyString_FromString PyUnicode_FromString
#endif


