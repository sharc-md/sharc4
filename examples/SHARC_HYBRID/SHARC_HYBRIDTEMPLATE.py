import datetime
import os

import yaml
from SHARC_HYBRID import SHARC_HYBRID

__all__ = ["SHARC_HYBEX"]

AUTHORS = "Author Authorson"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2025, 7, 24)
NAME = "HYBEX"
DESCRIPTION = "   HYBRID example interface"
CHANGELOGSTRING = ""


class SHARC_HYBEX(SHARC_HYBRID):
    """
    Minimal code for hybrid interfaces
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define attributes here

        # Define template keys and types here
        self.QMin.template.update({"children": None})
        self.QMin.template.types.update({"children": dict})

        # Simple template for child interfaces
        # See read_template
        self._child_template = {
            "interface": str,  # Name of the SHARC interface
            "args": list,  # Arguments for child interface
            "kwargs": dict,  # Keyword arguments for child interface
        }

    @staticmethod
    def description():
        return SHARC_HYBEX._description

    @staticmethod
    def version():
        return SHARC_HYBEX._version

    @staticmethod
    def name() -> str:
        return SHARC_HYBEX._name

    @staticmethod
    def versiondate():
        return SHARC_HYBEX._versiondate

    @staticmethod
    def changelogstring():
        return SHARC_HYBEX._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_HYBEX._authors

    def read_resources(self, resources_file="HYBEX.resources", kw_whitelist=None):
        super().read_resources(resources_file, kw_whitelist)
        # read_resources need to be defined here with the correct
        # name for the resources_file parameter for the interface.
        # No custom resource keys are defined for this example interface.

    def read_template(self, template_file="HYBEX.template", kw_whitelist=None):
        # It is recommended to use yaml format for hybrid templates.
        # Especially for multi child hybrids it simplifies template parsing
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)

            # Check if "children" dict is in template and if all keys are present
            # and of correct type. This is just an example of how a template file
            # for a multi child hybrid interface can look like.
            if "children" in tmpl_dict:
                for parameters in tmpl_dict["children"].values():
                    # Check all entries of children if the name of a SHARC interface
                    # is specified and if arguments and keyword arguments are defined
                    # in the template.
                    for k, v in self._child_template.items():
                        if k not in parameters or not isinstance(tmpl_dict[k], v):
                            self.log.error("Raise some error")
                            raise ValueError
            else:
                self.log.error("No children defined in template.")
                raise ValueError

        # When the checks passed, the yaml dictionary can be asigned to QMin
        self.QMin.template["children"] = tmpl_dict["children"]
        # Indicate that read_template was called. This has to be done if
        # super().read_template() from SHARC_INTERFACE is not called.
        self._read_template = True

    def setup_interface(self):
        super().setup_interface()

        # Here we initialize our child interfaces

        # First we define our local kindergarden dict.
        # The values have to be tuples.
        kindergarden = {
            name: (child["interface"], child["args"], child["kwargs"]) for name, child in self.QMin.template["children"]
        }
        # Then our kindergarden (self._kindergarden) is initialized
        self.instantiate_children(kindergarden)

        # Now we can setup our children
        for name, child in self._kindergarden.items():
            child.setup_mol(self.QMin)
            child.read_resources()
            child.read_template()
            child.setup_interface()

            # In general it is good practice to modify the scratch, pwd and save
            # directories for each child, although it is not mandatory
            child.QMin.resources["scratchdir"] = os.path.join(self.QMin.resources["scratchdir"], name)
            child.QMin.resources["pwd"] = os.path.join(self.QMin.resources["pwd"], name)
            child.QMin.resources["cwd"] = os.path.join(self.QMin.resources["cwd"], name)
            child.QMin.save["savedir"] = os.path.join(self.QMin.save["save"], name)


    def set_coords(self, xyz, pc = False):
        super().set_coords(xyz, pc)

        # Here we define the coordinates for each child
        # in this example every child gets the same.
        for child in self._kindergarden.values():
            child.set_coords(xyz, pc)

    def read_requests(self, requests_file = "QM.in"):
        super().read_requests(requests_file)

        # Here we define the requests for each child
        # in this example every child gets the same.
        for child in self._kindergarden.values():
            child.read_requests(requests_file)

    def run(self):
        # Now we execute our children
        # Note: run_children will execute the run() and getQMout() methods for each child
        self.run_children(self.log, self._kindergarden, self.QMin.resources["ncpu"])

    def getQMout(self):
        # In this example,the unmodified QMout from the first child
        # is returned.
        self.QMout = next(iter(self._kindergarden.values())).QMout
        return self.QMout

    def get_features(self, KEYSTROKES = None) -> set:
        # Here the features of the hybrid interface are defined.
        # The features may depend on the features of the children.
        return set()

    def get_infos(self, INFOS, KEYSTROKES = None):
        # Setup routine, see ab-initio example

    def prepare(self, INFOS, dir_path):
        # Setup routine, see ab.initio example