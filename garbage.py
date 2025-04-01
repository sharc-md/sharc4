    def _check_zmn(self, zmn_dict: dict[str, int]) -> bool:
        """
        Check if dictionary contains Z, M, N keys and validate if values are int
        """
        if not isinstance(zmn_dict, dict) or zmn_dict.keys() != {"Z", "M", "N"}:
            return False
        if not all(isinstance(val, int) for val in zmn_dict.values()):
            return False
        return True








            # Check whether all aufbau_site_states are in the site_states:
            for s in key["aufbau_site_states"]:
                message = f"Aufbau site state {s} of fragmnet {frag} is not included in the site states of the fragment!"
                if not s["Z"] in key["site_states"]:
                    self.log.error(message)
                    raise ValueError()
                if len(key["site_states"][s["Z"]]) < s["M"]:
                    self.log.error(message)
                    raise ValueError()
                if key["site_states"][s["Z"]][s["M"]-1] < s["N"]:
                    self.log.error(message)
                    raise ValueError()

        # Check whether all fragments have needed specifications for all full-system charges
        for f1 in self.QMin.template['fragments']:
            for f2 in self.QMin.template['fragments']:
                if set(CHARGES[f1]) != set(CHARGES[f2]):
                    self.log.error(
                            f"Not all fragments have refcharge and embedding_site_state dictionaries specified for the all full-system charges!"
                            )
                    raise ValueError()

        # Create charges atribute and check whether all requested charges are there
        # The later part is gonna be moved to read_requests function once charges are upgraded to the master level
        self.charges = CHARGES[next(iter(self.QMin.template['fragments']))]
        self.charges_to_do = set()
        for M, N in enumerate(self.QMin.molecule['states']):
            if N > 0:
                print(M)
                if not self.QMin.molecule['charge'][M] in self.charges:
                    self.log.error(
                            f"Requested full-system charge {C} cannot be calculated because refcharge and embedding_site_state of any fragmnet are not define for it!"
                            )
                    raise ValueError()
                else:
                    self.charges_to_do.add(self.QMin.molecule['charge'][M])

        for C in self.charges: 
            self.EHFjobs[C] = None
            self.ECIjobs[C] = None

