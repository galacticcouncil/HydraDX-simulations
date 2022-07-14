# TE Academy Details

## Course Description - Lectures

### Part One: Fundamentals
#### Session1: Introduction to the Hydra AMM (Kris)
- What is the Hydra AMM and how does it stand out from traditional AMMs?
- Distinguishing between various subsystems with core focus on the liquidity subsystem and its functionality
- Identifying the mechanisms of the liquidity subsystem and their connection to the agent roles and incentives via observable metrics.


#### Session 2: The Engineering Design Process: Overview (Barlin)
- What is a system?
- System design
- The Hydra AMM liquidity subsystem
- Thursday, Aug 26, 5pm-7pm CEST

#### Session 3: A Brief Introduction to Generalized Dynamical Systems (Jamsheed)
- What is a generalized dynamical system (GDS)?
- The state representation
- Actions, policies and behaviors
- The Hydra AMM as a GDS

#### Session 4: The Mathematical Specification (Barlin)
- What is a mathematical specification?
- Requirements, states and mechanisms
- The Hydra AMM math spec

#### Session 5: The Reference Implementation: cadCAD (Jamsheed/Barlin)
- What is cadCAD?
- Encoding a GDS in cadCAD
- Parameterization in cadCAD
- The Hydra AMM in cadCAD

#### Session 6: Parameter Selection Under Uncertainty (Jamsheed)
- The Parameter selection under uncertainty (PSuU) workflow
- Applying the PSuU workflow to the Hydra AMM cadCAD implementation
- A fee mechanism as a potential research project

### Part Two: Research Projects
[Sessions 7-8: Possible ‘slack’ for Part One; otherwise, commencement of learner-proposed research projects (cf. below)]

#### Sessions 7 - 12: Breakouts / hackathon style progress in small groups, each working on a proposed project direction

Instructors act as facilitators/mentors, reminding/calling back to methodology, concepts and/or algorithms presented in Part One that help unblock learners & provide efficiency gains in the progression from research topic to code implementation
Open-ended: no requirement to complete a research direction in the course; rather, learning objective is to be able to understand how to leverage Part One to rapidly make future progress on the research topic from Part Two
Hack & Share



## Reading Material

[Reading List](https://hackmd.io/2hFUaRRiT72VzxaB8yWwEA?view)

# cadCAD

```
                  ___________    ____
  ________ __ ___/ / ____/   |  / __ \
 / ___/ __` / __  / /   / /| | / / / /
/ /__/ /_/ / /_/ / /___/ ___ |/ /_/ /
\___/\__,_/\__,_/\____/_/  |_/_____/
by cadCAD                  ver. 0.4.23
======================================
       Complex Adaptive Dynamics       
       o       i        e
       m       d        s
       p       e        i
       u       d        g
       t                n
       e
       r
```
***cadCAD*** is a Python package that assists in the processes of designing, testing and validating complex systems 
through simulation, with support for Monte Carlo methods, A/B testing and parameter sweeping. 

# Getting Started


#### Change Log: [ver. 0.4.23](CHANGELOG.md)

[Previous Stable Release (No Longer Supported)](https://github.com/cadCAD-org/cadCAD/tree/b9cc6b2e4af15d6361d60d6ec059246ab8fbf6da)

## 0. Pre-installation Virtual Environments with [`venv`](https://docs.python.org/3/library/venv.html) (Optional):
If you wish to create an easy to use virtual environment to install cadCAD inside of, please use the built in `venv` package.

***Create** a virtual environment:*
```bash
$ python3 -m venv ~/cadcad
```

***Activate** an existing virtual environment:*
```bash
$ source ~/cadcad/bin/activate
(cadcad) $
```

***Deactivate** virtual environment:*
```bash
(cadcad) $ deactivate
$
```

## 1. Installation: 
Requires [>= Python 3.6](https://www.python.org/downloads/) 

**Option A: Install Using [pip](https://pypi.org/project/cadCAD/)** 
```bash
$ pip3 install cadCAD
```

**Option B:** Build From Source
```
$ pip3 install -r requirements.txt
$ python3 setup.py sdist bdist_wheel
$ pip3 install dist/*.whl
```

**Option C: Using [Nix](https://nixos.org/nix/)**
1. Run `curl -L https://nixos.org/nix/install | sh` or install Nix via system package manager
2. Run `nix-shell` to enter into a development environment, `nix-build` to build project from source, and 
`nix-env -if default.nix` to install

The above steps will enter you into a Nix development environment, with all package requirements for development of and 
with cadCAD. 

This works with just about all Unix systems as well as MacOS, for pure reproducible builds that don't 
affect your local environment.

## 2. Documentation:
* [Simulation Configuration](documentation/README.md)
* [Simulation Execution](documentation/Simulation_Execution.md)
* [Policy Aggregation](documentation/Policy_Aggregation.md)
* [Parameter Sweep](documentation/System_Model_Parameter_Sweep.md)
* [Display System Model Configurations](documentation/System_Configuration.md)

## 3. Connect:
Find other cadCAD users at our [Discourse](https://community.cadcad.org/). We are a small but rapidly growing community.

## 4. [Contribute!](CONTRIBUTING.md)