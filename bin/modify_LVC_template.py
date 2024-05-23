#!/usr/bin/env python3
from optparse import OptionParser
import numpy as np


def main(file, states, modes, no_trans_mult=False, no_es2es_trans_mult=False, no_es2es_trans_mult_for_mult=None):
    # read the V0-file
    new_template = []
    with open(file) as f:
        V0txt = f.readline()[:-1]
        new_template.append(V0txt)
        # get number of modes
        with open(V0txt, "r") as V0:
            line = V0.readline()
            while "Frequencies" not in line:
                line = V0.readline()
            nmodes = len(V0.readline().split())

        # perform mode selection
        selected_modes = []
        if modes != "all":
            try:
                for i in modes.split(","):
                    if "~" in i:
                        q = i.split("~")
                        for j in range(int(q[0]), int(q[1]) + 1):
                            selected_modes.append(j)
                    else:
                        selected_modes.append(int(i))
            except ValueError:
                print('Please enter integers or ranges of integers (e.g. "-3~-1  2  5~7")!')
            if any(m > nmodes for m in selected_modes):
                raise ValueError(f"Selected modes are exceed number of modes in {V0txt}: {nmodes}")
        else:
            selected_modes = list(range(1, nmodes + 1))
        selected_modes = set(selected_modes)

        # perform state selection
        template_states = [int(s) for s in f.readline().split()]
        if states == "all":
            new_template.append(" ".join(map(str, template_states)))
            states = template_states
        else:
            new_template.append(states)
            states = [int(s) for s in states.split()]

        if (len(states) > len(template_states)) or any(a > b for (a, b) in zip(states, template_states)):
            raise ValueError(f"{states} not compatible with {template_states} from template file!")

        selected_states = {(im + 1, s + 1) for im, ns in enumerate(states) for s in range(ns) if ns != 0}

        line = f.readline()
        if line == "epsilon\n":
            new_template.append("epsilon")
            selected = []
            z = int(f.readline()[:-1])

            def a(x):
                v = f.readline().split()
                return (int(v[0]), int(v[1]), v[2])

            for im, s, v in map(a, range(z)):
                if (im, s) in selected_states:
                    selected.append(f"{im:3d} {s:3d}  {v}")
            new_template.append(f"{len(selected)}")
            new_template.extend(selected)
            line = f.readline()

        if line == "eta\n":
            z = int(f.readline()[:-1])
            new_template.append("eta")
            selected = []
            for im, si, sj, v in map(
                lambda v: (int(v[0]), int(v[1]), int(v[2]), float(v[3])),
                map(lambda _: f.readline().split(), range(z)),
            ):
                if (im, si) in selected_states and (im, sj) in selected_states:
                    selected.append(f"{im:3d} {si:3d} {sj:3d} {v: .5e}")
            new_template.append(f"{len(selected)}")
            new_template.extend(selected)

            line = f.readline()

        if line == "kappa\n":
            new_template.append("kappa")
            selected = []
            z = int(f.readline()[:-1])

            def b(_):
                v = f.readline().split()
                return (int(v[0]), int(v[1]), int(v[2]), float(v[3]))

            for im, s, i, v in map(b, range(z)):
                if (im, s) in selected_states and i in selected_modes:
                    selected.append(f"{im:3d} {s:3d} {i:5d} {v: .5e}")
            new_template.append(f"{len(selected)}")
            new_template.extend(selected)

            line = f.readline()

        if line == "lambda\n":
            new_template.append("lambda")
            selected = []
            z = int(f.readline()[:-1])

            def c(_):
                v = f.readline().split()
                return (int(v[0]), int(v[1]), int(v[2]), int(v[3]), float(v[4]))

            for im, si, sj, i, v in map(c, range(z)):
                if (im, si) in selected_states and (im, sj) in selected_states and i in selected_modes:
                    selected.append(f"{im:3d} {si:3d} {sj:3d} {i:3d} {v: .5e}")
            new_template.append(f"{len(selected)}")
            new_template.extend(selected)
            line = f.readline()

        if line == "gamma\n":
            new_template.append("gamma")
            selected = []
            z = int(f.readline()[:-1])

            def d(_):
                v = f.readline().split()
                return (int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]), float(v[5]))

            for im, si, sj, n, m, v in map(d, range(z)):
                if (im, si) in selected_states and (im, sj) in selected_states and n in selected_modes and m in selected_modes:
                    selected.append(f"{im:3d} {si:3d} {sj:3d} {n:3d} {m:3d} {v: .7e}")
            new_template.append(f"{len(selected)}")
            new_template.extend(selected)
            line = f.readline()

        # get indices of the selected states in the old state vector
        template_nmstates = sum((im + 1) * s for im, s in enumerate(template_states))
        all = [(im + 1, s + 1) for im, ns in enumerate(template_states) for s in range((im + 1) * ns)]
        selected_nmstates = [(im + 1, s + 1) for im, ns in enumerate(states) for s in range((im + 1) * ns)]
        idx = [i for i, (im, s) in enumerate(all) if (im, s) in selected_nmstates]

        while line:
            if "SOC" in line:
                new_template.append(line[:-1])
                mat = np.zeros((template_nmstates, template_nmstates), dtype=float)
                line = f.readline()
                i = 0
                while len(line.split()) == template_nmstates:
                    mat[i, :] += np.asarray(line.split(), dtype=float)
                    i += 1
                    line = f.readline()
                mat = mat[idx, :][:, idx]
                new_template.extend(["".join(map(lambda x: f" {x: .7e}", row)) for row in mat])

            elif line[:2] == "DM":
                new_template.append(line[:-1])
                mat = np.zeros((template_nmstates, template_nmstates), dtype=float)
                line = f.readline()
                i = 0
                while len(line.split()) == template_nmstates:
                    mat[i, :] += np.asarray(line.split(), dtype=float)
                    i += 1
                    line = f.readline()
                mat = mat[idx, :][:, idx]
                new_template.extend(["".join(map(lambda x: f" {x: .7e}", row)) for row in mat])

            elif "Multipolar Density Fit" in line:
                new_template.append(line[:-1])
                line = f.readline()
                n_fits = int(line)
                selected = []

                def d(_):
                    v = f.readline().strip().split(maxsplit=4)
                    return (int(v[0]), int(v[1]), int(v[2]), int(v[3]), v[4])

                # get first element in states
                gs = next(map(lambda x: (x[0] + 1, 1), filter(lambda x: x[1] != 0, enumerate(template_states))))

                for im, si, sj, i, v in map(d, range(n_fits)):
                    if no_es2es_trans_mult and (im, si) != gs and (im, sj) != gs and si != sj:
                        continue
                    elif no_trans_mult and si != sj:
                        continue
                    elif (
                        no_es2es_trans_mult_for_mult is not None
                        and im == no_es2es_trans_mult_for_mult
                        and (im, si) != gs
                        and (im, sj) != gs
                        and si != sj
                    ):
                        continue
                    if (im, si) in selected_states and (im, sj) in selected_states:
                        if v[0] != "-":
                            v = " " + v
                        selected.append(f"{im} {si:2} {sj:2} {i:3}     {v}")
                new_template.append(f"{len(selected)}")
                new_template.extend(selected)
            else:
                line = f.readline()
        f.close()

    # write the file to stdout
    print("\n".join(new_template))


if __name__ == "__main__":
    parser = OptionParser()
    parser.set_usage(
        """
============================================================================
                            modify LVC-template

                        author: Severin Polonius
============================================================================

usage: python3 {sys.argv[0]} -s='<states>' -m='<modes>' LVC.template > LVC_mod.template

    states is a string e.g.: '2 0 2' for 2 Singlets and 2 Triplets
    modes is a string e.g.: '7~12,14,15~89' (range expressions allowed)
"""
    )
    parser.add_option("-s", "--states", dest="states", type="str", default="all", help="specify the states")
    parser.add_option("-m", "--modes", dest="modes", type="str", default="all", help="specify the modes")
    parser.add_option(
        "--no-transition-multipoles",
        dest="no_trans_mult",
        action="store_true",
        default=False,
        help="exclude transition multipoles (s1!=s2) from template",
    )
    parser.add_option(
        "--no-es2es-transition-multipoles",
        dest="no_es2es_trans_mult",
        action="store_true",
        default=False,
        help="exclude transition multipoles (s1 != 0 and s1!=s2) from template",
    )
    parser.add_option(
        "--no-es2es-transition-multipoles-for-mult",
        dest="no_es2es_trans_mult_for_mult",
        type=int,
        default=None,
        help="exclude transition multipoles (s1 != 0 and s1!=s2) for multiplicity from template",
    )

    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.print_usage()
    main(
        args[0],
        options.states,
        options.modes,
        options.no_trans_mult,
        options.no_es2es_trans_mult,
        options.no_es2es_trans_mult_for_mult,
    )
