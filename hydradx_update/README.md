# Modeling Omnipool in cadCAD
- TestSwap.ipynb demonstrates how cadCAD can be run
- The "spec" folder has the mathematical spec of the implemented AMM
- 
- 14 Dec 2021
- New omnipool-reweighting model with parameter "a"
- It is possible to switch and run everything using omnipool-uni model
- Log for debugging, switched off by default 
- No need for any changes to run tests, just run "test_reweighting_amm.py" 
- If you need to run CAD CAD simulation using "TestSwap.ipynb" or "TestHDXSwap.ipynb", please select model in "select_model.txt"

- 23 Dec 2021
- New withdrawal mechanics
- Optional logs for swaps, liquidity add liquidity  withdrawal

- 16 Jan 2021
- New IL mechanics
- New price formula for reweighting amm
- New optional logs for IL and other important parts of code
- Updated tests
- New testing files for reweighting amm: TestHDXSwap_reweighting.ipynb, TestIL_reweighting.ipynb, TestIL_reweighting.ipynb, TestHDXSwap_reweighting.ipynb