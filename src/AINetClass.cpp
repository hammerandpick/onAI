#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <climits>
#include <algorithm>    // std::random_shuffle
#include <vector>
#include <time.h>
#include <mutex>
#include "AINetClass.h"

AINetClass::AINetClass()
{
	this->iNumInputNodes = 2;
	this->iNumOutputNodes = 1;
	this->vdNetworkTopology.clear();
	this->vdNetworkTopology.resize(3,1);
	this->vdNetworkTopology.at(0) = this->iNumInputNodes;
	this->vdNetworkTopology.at(1) = 2;
	this->vdNetworkTopology.at(2) = this->iNumOutputNodes;
	this->iMaxIterations = 1000;
	this->dlearningRate = 0.2;
	this->iCounter = 0;
	this->iTrainingDataRowsMax = 0;
	this->iActivationFunction = 0;
	this->bOptionShuffle = false;
	this->bOptionIO = false;
	this->vTrainingDataColumns.resize(this->iNumInputNodes);
	this->vTrainingDataColumns.clear();
	this->vecValues.clear();
	this->vecWeights.clear();
	this->vecThresholds.clear();
	this->vecExpectedValues.clear();
	this->vecCalcDelta.clear();
	this->errorList.clear();
	this->vvErrors.clear();
	this->iTimeNumInputColumns = 0;
	this->iTimePreviousRows = 0;
	this->bOptionMaxIterationSet = false;
	this->bOptionAutoGenerate = false;
	this->bOptionDisplayAllNodes = false;

}

AINetClass::~AINetClass()
{
	this->vecExpectedValues.clear();
	this->vecValues.clear();
	this->vecWeights.clear();
	this->vecThresholds.clear();
	this->vvTrainingDataMatrix.clear();
	this->vecCalcDelta.clear();
}

unsigned int AINetClass::NUMNODES()
{
	// returns number of nodes
	unsigned int totalNumberNodes = 0;
	totalNumberNodes += this->NUMINPUTNODES();
	for (unsigned int currentLayer = 2; currentLayer <= 1 + this->getNumberOfLayers(true); currentLayer++)
	{
		totalNumberNodes += this->getNumberOfNodesInLayer(currentLayer);
	}
	totalNumberNodes += this->NUMOUTPUTNODES();
	return totalNumberNodes;
}

unsigned int AINetClass::NUMINPUTNODES()
{
	//returns number of input nodes
	return this->getNumberOfInputNodes();
}

unsigned int AINetClass::NUMREALINPUTNODES()
{
	//returns number of real input nodes;
	return this->iNumRealInputNodes;
}

unsigned int AINetClass::NUMOUTPUTNODES()
{
	// returns number of output nodes
	return this->getNumberOfOutputNodes();
}

unsigned int AINetClass::NUMHIDDENNODES()
{
	// return number of hidden nodes
	unsigned int iCalc = 0;
	for (unsigned int i = 1;i<this->vdNetworkTopology.size()-1;i++)
	{ 
		iCalc += this->vdNetworkTopology.at(i);
	}
	return iCalc;
}

unsigned int AINetClass::SizeOfArray()
{
	return 1+this->NUMNODES();
}

unsigned int AINetClass::getMaxIterations()
{
	// returns the number of maxIterations
	return iMaxIterations;
}

unsigned int AINetClass::Counter(bool bIncrease )
{
	if (bIncrease)
	{
		this->iCounter += 1;
		// now shuffle the list
		if ((this->iCounter % this->getTrainingDataRowsMax() == 0) && bOptionShuffle)
		{
			this->shuffleTrainingData();
		}
	}
	return this->iCounter;
}

unsigned int AINetClass::CurrentTrainingDataRow()
{
	// returns current Training Data Row
	unsigned int tmpReturn;
	unsigned int tmpMaxRows = this->getTrainingDataRowsMax();
	if (tmpMaxRows == 0)
	{
		this->throwFailure("division by 0 iTrainingDataRowsUseMax", true);
		tmpReturn= 0;
	}
	else
	{
		unsigned int iDiv = this->iCounter % tmpMaxRows;
		// pull the date from the list.
		if (iDiv < this->inputDataPullList.size())
		{
			tmpReturn = this->inputDataPullList.at(iDiv);
		}
		else
		{
			this->throwFailure("pull list exeeded.", true);
		}
	}
	return tmpReturn;
}

unsigned int AINetClass::getActivationFunction()
{
	return this->iActivationFunction;
}

unsigned int AINetClass::getActivationFunction(unsigned int tmpNodeID)
{
	// this function returns the correct activation function for requested node
	unsigned int tmpLayer = this->getLayerByNode(tmpNodeID);
	return this->viLayerActivationFunction.at(min(tmpNodeID, this->viLayerActivationFunction.size() - 1));
}

unsigned int AINetClass::getNumberOfNodesInLayer(int tmpLayer)
{
	// get the number of nodes in specified layer.
	// if tmpLayer is negative it is fetched in reverse order. so output layer is -1
	unsigned int tmpReturn = 0;

	this->validLayer(tmpLayer);
	tmpReturn = this->vdNetworkTopology.at(max(0, tmpLayer - 1));
	return tmpReturn;
}

unsigned int AINetClass::getNumberOfLayers(bool bOnlyHidden)
{
	// returns the number of layers
	if (bOnlyHidden) return(unsigned int) this->vdNetworkTopology.size()-2;// removing input and output layer
	else return (unsigned int) this->vdNetworkTopology.size();
}

unsigned int AINetClass::getLayerStart(int tmpLayer, bool falseForLayerEnd)
{
	// returns begin of layer (or end of layer if parameter is false.
	int chosenLayer = 0;
	unsigned int retInt = 0;
	chosenLayer = this->validLayer(tmpLayer) - 1;

	if (chosenLayer == 0 && falseForLayerEnd)
	{
		retInt = 1;
	}
	else
	{
		for (int i = 0; i <= chosenLayer; i++)
		{
			if (i < chosenLayer)
			{
				// sum all previous layers
				retInt += this->vdNetworkTopology.at(i);
			}
			else
			{
				// add one for begin or the number of nodes for end
				if (falseForLayerEnd)
				{
					retInt += 1; // begin
				}
				else
				{
					// count 
					retInt += this->vdNetworkTopology.at(i);
				}
				break;
			}
		}
	}
	return retInt;
}

double AINetClass::LearningRate()
{
	// returns current/selected learning rate
	return dlearningRate;
}

size_t AINetClass::TrainingDataColumns()
{
	// this one returns the number of columns filled with names
	return vTrainingDataColumns.size();
}

std::string AINetClass::TrainingDataColumnName(unsigned int tmpColumn, bool shortList)
{
	// return the Name of the DataColumn
	std::string tmpString="no column name";
	std::vector<unsigned int> retPullList;
	if (!shortList)
	{
		// generate List 
		if (this->iTimeNumInputColumns == 0)
		{
			// repeated inputs
			retPullList.resize(1 + this->getNumberOfInputNodes() + this->getNumberOfOutputNodes(),0);
			for (unsigned int i = 0; i < retPullList.size(); i++)
			{
				if (i <= this->iNumRealInputNodes)
				{
					retPullList.at(i) = i;
				}
				else
				{
					retPullList.at(i) = i % this->iNumRealInputNodes+1;
				}
			}
		}
		else
		{
			// repeated inputs
			retPullList.resize(1 + this->getNumberOfInputNodes() + this->getNumberOfOutputNodes(),0);
			for (unsigned int i = 0; i < retPullList.size(); i++)
			{
				if (i <= this->iNumRealInputNodes)
				{
					retPullList.at(i) = i;
				}
				else
				{
					if ((i - this->iNumRealInputNodes) % (this->iTimeNumInputColumns) == 0)
					{
						retPullList.at(i) = this->iTimeNumInputColumns;
					}
					else
					{
						retPullList.at(i) = (i - this->iNumRealInputNodes) % (this->iTimeNumInputColumns);
					}
				}
			}
		}
		tmpColumn = retPullList.at(tmpColumn);
	}
	if ((tmpColumn < vTrainingDataColumns.size()) && (tmpColumn >= 0))
	{
		tmpString = vTrainingDataColumns.at(tmpColumn);
	}
	return tmpString;
}

unsigned int AINetClass::getTrainingDataRowsMax()
{
	//returns number of training data rows
	/* this one calculates the number of training data columns
	/ which is total number of rows reduced by
	/	- previous Data (historical)
	/	- next Data (historical)
	/	- portion of verification data
	*/
	return max(1,this->iTrainingDataRowsMax - abs(this->iTimePreviousRows));
}

unsigned int AINetClass::getMaximumNodesLayer(bool bGetMaximumNodes)
{
	// returns number of layer layer with max nodes or maximum number of nodes in a layer (if true is set)
	unsigned int retValue = this->NUMINPUTNODES();
	unsigned int retLayer = 0;
	for (unsigned int currentLayer = 0; currentLayer < this->getNumberOfLayers();currentLayer++)
	{
		retValue = max(retValue, this->vdNetworkTopology.at(currentLayer));
		if (retValue == this->vdNetworkTopology.at(currentLayer))
		{
			retLayer = currentLayer+1;
		}
	}
	
	if (bGetMaximumNodes)
		return retValue;
	else
		return retLayer;
}

unsigned int AINetClass::getLayerByNode(unsigned int iTmpNode)
{
	// this function returns the corresponding layer for a specific node
	unsigned int returnLayer = 0;
	unsigned int LayerBegin = 1;
	for (unsigned int i = 0; i < this->vdNetworkTopology.size(); i++)
	{
		if ((LayerBegin <= iTmpNode) && (iTmpNode < LayerBegin + this->vdNetworkTopology.at(i)))
		{
			// this is the correct layer
			returnLayer = i;
			break;
		}
		else
		{
			LayerBegin += this->vdNetworkTopology.at(i);
		}
	}
	return returnLayer;
}

bool AINetClass::continueCalculation()
{
	// continue calculation
	bool bContCalc = false;
	if (this->Counter() < this->getMaxIterations())
	{
		bContCalc=true;
	}
	return bContCalc;
}

bool AINetClass::IsTrainingRestart()
{
	// returns true if training data has restarted
	if (iCounter % this->getTrainingDataRowsMax() == 0)
		return true;
	else 
		return false;
}

bool AINetClass::IsLastLayer(unsigned int tmpLayer)
{
	if (tmpLayer == this->vdNetworkTopology.size())
		return true;
	else
		return false;
}

bool AINetClass::getOptionStatus()
{
	// returns option status
	return this->bOptionStatus;
}

bool AINetClass::setNumInputNodes(unsigned int tmpInputNodes)
{
	// set the number of input nodes
	this->initializationDone = false;		// significant changes to network, so initialization should be renewed
	this->iNumInputNodes = min(max(1, tmpInputNodes), UINT_MAX);
	this->vdNetworkTopology.at(0) = this->iNumInputNodes;
	this->iNumRealInputNodes = this->iNumInputNodes;
	this->resizeVectors();
	return (iNumInputNodes == tmpInputNodes);
}

bool AINetClass::setNumOutputNodes(unsigned int tmpOutputNodes)
{
	// set the number of output nodes
	this->initializationDone = false;		// significant changes to network, so initialization should be renewed
	this->iNumOutputNodes = min(max(1, tmpOutputNodes), UINT_MAX);
	this->vdNetworkTopology.at(this->vdNetworkTopology.size() - 1) = this->iNumOutputNodes;
	return (iNumOutputNodes == tmpOutputNodes);
}

bool AINetClass::setTimePrevRows(int tmpPrevRows)
{
	// set the number of rows to be used as memorized input data.
	this->initializationDone = false;		// significant changes to network, so initialization should be renewed
	this->iTimePreviousRows = abs(tmpPrevRows);
	if (tmpPrevRows == 0)
	{
		bHistoricData = false;
	}
	else
	{
		bHistoricData = true;
		if (this->iTimeNumInputColumns == 0)
		{
			this->iNumInputNodes = this->iNumRealInputNodes * (1 + this->iTimePreviousRows);
		}
		else
		{
			this->iNumInputNodes = this->iNumRealInputNodes + this->iTimeNumInputColumns * this->iTimePreviousRows;
		}
	}
	this->resizeVectors();
	this->recalculateInputDataPullList();
	return (iTimePreviousRows == tmpPrevRows);
}

bool AINetClass::setTimeInputColumns(unsigned int tmpPrevCols)
{
	// set the number of input columns
	this->initializationDone = false;		// significant changes to network, so initialization should be renewed
	if (this->iTimePreviousRows == 0)
	{
		bHistoricData = false;
	}
	else
	{
		bHistoricData = true;
		this->iNumInputNodes = this->iNumRealInputNodes + this->iTimePreviousRows * tmpPrevCols;
	}
	this->iTimeNumInputColumns = min(max(0, tmpPrevCols), this->iNumRealInputNodes);
	this->resizeVectors();
	return (iTimeNumInputColumns == tmpPrevCols);
}

bool AINetClass::setMaxIterations(unsigned int tmpMaxIterations)
{
	// SetMaximumNumber of iterations
	this->bOptionMaxIterationSet = true;
	this->iMaxIterations = min(max(1, tmpMaxIterations), UINT_MAX);
	return (this->iMaxIterations == tmpMaxIterations);
}

bool AINetClass::setLearningRate(double tmpLearningRate)
{
	// set the learning rate
	dlearningRate = min(max(0,tmpLearningRate),500);
	return (dlearningRate == tmpLearningRate);
}

bool AINetClass::setTrainingDataRowsMax(unsigned int tmpMaxRows)
{
	// set the maximum number of training data rows
	if (tmpMaxRows > 0)
	{
		this->iTrainingDataRowsMax = min(max(1, tmpMaxRows), UINT_MAX);
	}
	else
	{
		initializationDone = false;
	}
	this->recalculateInputDataPullList();
	return (this->iTrainingDataRowsMax == tmpMaxRows);
}

bool AINetClass::setNumberOfHiddenLayers(unsigned int tmpHiddenLayers, unsigned int tmpNodesinHiddenLayer)
{
	// set the number of hidden layers
	// initialization required
	this->initializationDone = false;
	if (this->optionNoDeep)
	{
		tmpHiddenLayers = 1;
	}
	this->vdNetworkTopology.resize(max(2, tmpHiddenLayers + 2), max(1,tmpNodesinHiddenLayer));
	this->viLayerActivationFunction.resize(max(2, tmpHiddenLayers + 2), 0);
	// set number of input nodes
	this->vdNetworkTopology.at(0) = this->iNumInputNodes;
	// set number of output nodes
	this->vdNetworkTopology.at(this->vdNetworkTopology.size() - 1) = this->iNumOutputNodes;
	return (this->vdNetworkTopology.capacity() == tmpHiddenLayers + 2);
}

bool AINetClass::setNumberOfNodesinLayer(int tmpLayer, unsigned int tmpNumberOfNodes)
{
	bool tmpReturn = false;
	this->validLayer(tmpLayer);
	// tmpLayer is in valid range
	this->vdNetworkTopology.at(max(0, tmpLayer - 1)) = max(1, tmpNumberOfNodes);
	return true;
}

bool AINetClass::resetCounter()
{
	this->iCounter = 0;
	return (this->iCounter == 0);
}

void AINetClass::TrainingDataColumnPush_Back(std::string tmpString)
{
	vTrainingDataColumns.push_back(tmpString);
}

void AINetClass::shuffleTrainingData()
{
	// shuffling Training Data
	std::random_shuffle(this->inputDataPullList.begin(), this->inputDataPullList.end());
}

void AINetClass::activateNetwork()
{
	// activate the network.

	unsigned int numHiddenLayers = this->getNumberOfLayers(true);

	for (unsigned int currentLayer = 2; currentLayer <= 1 + this->getNumberOfLayers(true); currentLayer++)
	{
		// do this for each internal layer
		for (unsigned int h = this->getLayerStart(currentLayer); h <= this->getLayerStart(currentLayer, false); h++)
		{
			// do this for each node in layer
			double weightedInput = 0.0;
			for (unsigned int p = this->getLayerStart(currentLayer - 1); p <= this->getLayerStart(currentLayer - 1, false); p++)
			{
				// do this for each node in previous layer
				weightedInput += this->vecWeights[p][h] * this->vecValues[p];
			}
			// handle the thresholds
			weightedInput += (-1 * this->vecThresholds[h]);
			this->vecValues[h] = this->NodeFunction(weightedInput, h);
		}
	}

	for (unsigned int o = this->getLayerStart(-1); o <= this->getLayerStart(-1, false); o++)
	{
		double weightedInput = 0.0;
		for (unsigned int d = this->getLayerStart(-2); d <= this->getLayerStart(-2, false); d++)
		{
			weightedInput += this->vecWeights[d][o] * this->vecValues[d];
		}
		// handle the thresholds
		weightedInput += (-1 * this->vecThresholds[o]);
		this->vecValues[o] = this->NodeFunction(weightedInput, o);
	}
}

void AINetClass::setOptionCSV(bool bSetGerman)
{
	// set option csv german
	this->bOptionCSVGER = bSetGerman;
}

void AINetClass::setOptionDisplayAllNodes(bool bDisplayAll)
{
	// setting option display all nodes
	this->bOptionDisplayAllNodes = bDisplayAll;
}

unsigned int AINetClass::LoadTrainingDataFile()
{
	// read training data file
	// local variables
	std::string theLine = "no open file.";
	int theFirstElement = 0;
	unsigned int iNumberOfLines = 0;
	unsigned int iNumberOfFalseLines = 0;
	int iTimePreviousElements = 0;	// How many previous rows for calculation?


	std::ifstream theAIDataFile;

	this->openTrainingDataFile(theAIDataFile);

	// clear old date
	this->vvTrainingDataMatrix.clear();
	this->vvErrors.clear();

	if (theAIDataFile.is_open())
	{
		// Read First Line and output for Information.
		std::getline(theAIDataFile, theLine);
		printf("\nloading data...\n[%s]\n", theLine.c_str());
		//read the second line and reconfigure network.
		std::getline(theAIDataFile, theLine);
		this->generateFileInput(theLine);
		if ((theLine.find_first_of(",") == theLine.npos))
		{
			// this is no german style csv but german style is set
			// converting line backwards
			this->generateFileOutput(theLine);
			// disabling option ger
			this->setOptionCSV(false);
		}
		this->createNetwork(this->splitString(theLine, ","));

		std::vector<double> vdLocalVector(1 + this->NUMREALINPUTNODES() + this->NUMOUTPUTNODES());
		std::vector<double> vdOutputDummy(this->NUMOUTPUTNODES());
		std::string loadedNumber = "";

		// now start looking for maxiterations in aidatafile
		// and setting maximum iterations
		std::getline(theAIDataFile, theLine);
		this->generateFileInput(theLine);
		theFirstElement = 0;
		int currentElementCounter = 0;
		while (theLine.length() > 0)
		{
			if (theLine.find_first_of(",") != theLine.npos) theFirstElement = (int)theLine.find_first_of(",");
			else theFirstElement = (int)theLine.length();
			// read value
			// example for this line:
			// 1000,-5,3
			// 1000 iterations, -5 5 line above current line are historic data, 3 columns are used for historic data (5x3=)15 additional input nodes added.
			switch (currentElementCounter)
			{
			case 0:
				// first one on this line is maxiterations
				if (!this->bOptionMaxIterationSet)
				{
					this->setMaxIterations(atoi(theLine.substr(0, theFirstElement).c_str()));
				}
				break;
			case 1:
				//second element on this line is number of elements in timescale
				this->setTimePrevRows(atoi(theLine.substr(0, theFirstElement).c_str()));
				break;
			case 2:
				this->setTimeInputColumns(atoi(theLine.substr(0, theFirstElement).c_str()));
				break;
			case 5:
				this->setPercentVerification(atof(theLine.substr(0, theFirstElement).c_str()));
				break;
			default:
				break;
			}
			// delete value from whole string
			theLine.erase(0, theFirstElement + 1);
			currentElementCounter += 1; // increase element counter
		}
		// one line for headers 
		std::getline(theAIDataFile, theLine);
		this->generateFileInput(theLine);
		theFirstElement = 0;
		this->TrainingDataColumnPush_Back("intentionally left blank");
		while (theLine.length() > 0)
		{
			if (theLine.find_first_of(",") != theLine.npos) theFirstElement = (int)theLine.find_first_of(",");
			else theFirstElement = (int)theLine.length();
			// read value
			this->TrainingDataColumnPush_Back(theLine.substr(0, theFirstElement));
			// delete value from whole string
			theLine.erase(0, theFirstElement + 1);
		}

		//resize the vector
		vdLocalVector.clear();
		vdLocalVector.resize(1 + this->NUMREALINPUTNODES() + this->NUMOUTPUTNODES());

		while (!theAIDataFile.eof())
		{
			// now begin to read data values
			std::getline(theAIDataFile, theLine);
			this->generateFileInput(theLine);
			// clear vector
			vdLocalVector.clear();
			theFirstElement = 0;
			vdLocalVector.push_back(1.0); // first element is base/threshold value and always ste to 1.0

										  // looking for first element
			if (theLine.find_first_of(",") != theLine.npos) theFirstElement = (int)theLine.find_first_of(",");
			else theFirstElement = (int)theLine.length();

			// clearing data from previous line
			loadedNumber = "";
			while ((theFirstElement > 0) && (vdLocalVector.size()<vdLocalVector.capacity())) // cancel if vector already has aincNetwork.NUMINPUTNODES() + aincNetwork.NUMOUTPUTNODES() +1 values
			{
				// read value
				loadedNumber = theLine.substr(0, theFirstElement);
				// delete value from whole string
				theLine.erase(0, theFirstElement + 1);
				vdLocalVector.push_back(strtod(loadedNumber.c_str(), NULL));

				// check for next column
				if (theLine.find_first_of(",") != theLine.npos) theFirstElement = (int)theLine.find_first_of(",");
				else theFirstElement = (int)theLine.length();
			}
			if (vdLocalVector.size() >= 1 + this->iNumRealInputNodes + this->NUMOUTPUTNODES())
			{
				// counting number of lines and copying whole row to vector<vector>
				this->vvTrainingDataMatrix.push_back(vdLocalVector);
				this->vvErrors.push_back(vdOutputDummy);
				iNumberOfLines += 1;
			}
			else
			{
				// counting false/erronous lines
				iNumberOfFalseLines += 1;
			}
		}
		this->setTrainingDataRowsMax(iNumberOfLines);
	}

	this->closeTrainingDataFile(theAIDataFile);

	return iNumberOfFalseLines;
}

void AINetClass::setOptionShuffle(bool bSetShuffle)
{
	// setOptionShuffle
	bOptionShuffle = bSetShuffle;
}

void AINetClass::setOptionSilent(bool bSilent)
{
	// setting option Silent e.g. for threaded operation
	this->bSilent = bSilent;
	this->bOptionDisplayAllNodes = !bSilent;
	this->bOptionIO = !bSilent;
}

void AINetClass::setOptionNoDeep(bool bSetNoDeep)
{
	// set option to prevent usage of deep network
	this->optionNoDeep = bSetNoDeep;
}

void AINetClass::setOptionIO(bool bSetIO)
{
	// changes status of option IO
	this->bOptionIO = bSetIO;
}

void AINetClass::setOptionWeight(bool bSetWeight)
{
	// setting option weight
	this->optionWeight = bSetWeight;
}

void AINetClass::setOptionStatus(bool bSetStatus)
{
	// settion option status
	this->bOptionStatus = bSetStatus;
}

void AINetClass::setOptionNodeFunction(unsigned int tmpNodeFunction)
{
	this->setActivationFunction(tmpNodeFunction);
}

void AINetClass::setOptionThreadCombinatingMode(unsigned int iTCMode)
{
	// set the option for splitting or combining threads
	this->iThreadedCombinationMode = min(10, max(0, iTCMode));
}

void AINetClass::setPercentVerification(double tmpPercentVerifiy)
{
	// setting the amount of verification data to be used
	if (tmpPercentVerifiy > 1)
	{
		tmpPercentVerifiy = tmpPercentVerifiy / 100.0;
	}
	this->dPercentVerification = min(0, max(0.9, tmpPercentVerifiy));
}

void AINetClass::setTrainingRow(unsigned int iTmpRow)
{
	// set the next training row 
	this->iCounter = min(this->getTrainingDataRowsMax(),max(0,iTmpRow));
	this->trainLine();
}

void AINetClass::setInputOffset(double tmpInputOffset)
{
	// set offset for all used input nodes
	this->dInputOffset = tmpInputOffset;
}

void AINetClass::sortNetwork()
{
	// this function sorts the network
	
	// begin sorting at the end of the network (of course we won't sort output)
	for (unsigned int iSort = this->getNumberOfLayers(); iSort > 1; iSort--)
	{
		// reload network variables to tmp variables each layer.
		std::vector<double> vdTmpValues = this->vecValues;
		std::vector<double> vdTmpExpectedValues = this->vecExpectedValues;
		std::vector<std::vector<double>> vdTmpWeights = this->vecWeights;
		std::vector<double> vdTmpThresholds = this->vecThresholds;
		std::vector<double> vdTmpCalcDelta = this->vecCalcDelta;
		printf("\nCurrent Layer %i", iSort);
		// Sum all the weights
		// to do that create a vector with sum
		std::vector<double> vdSumWeights(this->getNumberOfNodesInLayer(iSort - 1),0.0);
		std::vector<double> vdSumWeightsSorted(this->getNumberOfNodesInLayer(iSort - 1), 0.0);
		std::vector<unsigned int> viSortList(this->getNumberOfNodesInLayer(iSort - 1), 0);
		for (unsigned int iWeightLayer = this->getLayerStart(iSort - 1); iWeightLayer <= this->getLayerStart(iSort - 1, false); iWeightLayer++)
		{
			unsigned int i = iWeightLayer - this->getLayerStart(iSort - 1);
			// now get all the weights of the nodes in the previous layer
			for (unsigned int iNodeLayer = this->getLayerStart(iSort); iNodeLayer <= this->getLayerStart(iSort, false); iNodeLayer++)
			{
				// sum the weights (absolut)
				vdSumWeights.at(i) = vdSumWeights.at(i) + abs(this->vecWeights[iWeightLayer][iNodeLayer]);
			}
		}
		vdSumWeightsSorted = vdSumWeights;
		//sort
		std::sort(vdSumWeightsSorted.begin(), vdSumWeightsSorted.end());
		// now population sortlist
		
		for (unsigned int i = 0; i < vdSumWeightsSorted.size(); i++)
		{
			// start at minimum of vdSUmWeightsSorted
			for (unsigned int j = 0; j < vdSumWeights.size(); j++)
			{
				if (vdSumWeights.at(j) == vdSumWeightsSorted.at(i))
				{
					// found corresponding element
					viSortList.at(i) = j;
					// clearing sumweights at this point to prevent doubles being sorted incorrectly. calculation should never be -1 because it is caclulated as abs.
					vdSumWeights.at(j) = -1.0;
					break;
				}
			}
		}

		// prevent sorting of input layer
		if(iSort>2)
		{
			// crawl all the elements 
			unsigned int iBegin = this->getLayerStart(iSort - 1);
			unsigned int iNode = 0;
			for (unsigned int i = 0; i < this->getNumberOfNodesInLayer(iSort-1); i++)
			{
				iNode = i + iBegin;
				this->vecValues.at(iNode) = vdTmpValues.at(iBegin+viSortList.at(i));
				this->vecThresholds.at(iNode) = vdTmpThresholds.at(iBegin + viSortList.at(i));
				this->vecCalcDelta.at(iNode) = vdTmpCalcDelta.at(iBegin + viSortList.at(i));
				this->vecExpectedValues.at(iNode) = vdTmpExpectedValues.at(iBegin + viSortList.at(i));
				unsigned int jNode = 0;
				unsigned int jBegin = this->getLayerStart(iSort - 2);
				// sort weights to previous layer
				for (unsigned int j = 0; j < this->getNumberOfNodesInLayer(iSort - 2); j++)
				{
					jNode = jBegin + j;
					this->vecWeights.at(jNode).at(iNode) = vdTmpWeights[jNode][iBegin + viSortList.at(i)];
					//todo continue
				}
				// sort weights from previous layer
				jBegin = this->getLayerStart(iSort);
				for (unsigned int j = 0; j < this->getNumberOfNodesInLayer(iSort); j++)
				{
					jNode = jBegin + j;
					this->vecWeights.at(iNode).at(jNode) = vdTmpWeights[iBegin + viSortList.at(i)][jNode];
					//todo continue
				}
			}
		}
		// sorting is now finished
	}
}

void AINetClass::initialize()
{
	// this inititalizes the network with previously defined parameters
	unsigned int tmpTotalNumberNodes = this->NUMNODES();

	this->vecValues.clear();
	this->vecValues.reserve(tmpTotalNumberNodes);
	this->vecCalcDelta.clear();
	this->vecCalcDelta.reserve(tmpTotalNumberNodes);
	this->vecWeights.clear();
	this->vecWeights.reserve(tmpTotalNumberNodes);
	this->vecThresholds.clear();
	this->vecThresholds.reserve(tmpTotalNumberNodes);
	this->vecExpectedValues.clear();
	this->vecExpectedValues.reserve(tmpTotalNumberNodes);
	for (unsigned int i = 0; i <= tmpTotalNumberNodes; i++)
	{
		// even if vector <int> [0] is defined. it is not to be used.
		this->vecValues.push_back(0.0);
		this->vecCalcDelta.push_back(0.0);
		this->vecThresholds.push_back(0.0);
		this->vecExpectedValues.push_back(0.0);
		std::vector<double> tmpRow;
		for (unsigned int y = 0; y <= tmpTotalNumberNodes; y++) {

			tmpRow.push_back(0.0);
		}
		this->vecWeights.push_back(tmpRow);
	}
	this->initializationDone = true;
}

bool AINetClass::openTrainingDataFile(std::ifstream &ptrDataFile)
{
	// open training data file for reading training data
	ptrDataFile.open(this->theAIDataFileName.c_str());
	if (ptrDataFile.is_open())
	{
		// good file is open.
		return true;
	}
	else
	{
		printf("File %s couldn't be opened.\n", this->theAIDataFileName.c_str());
		return false;
	}
}

void AINetClass::saveResultingNetwork(unsigned int iNumber)
{
	// saving all data to a file
	unsigned int iMaxNodes = 0;
	std::string cResultingNetworkFileName;
	cResultingNetworkFileName = "-" + std::to_string(iNumber) + "-results-ainetwork.csv";
	iMaxNodes = (unsigned int)this->vecWeights.size();
	// output Weight to file
	std::ofstream fileResultingNetwork;
	time_t t = time(NULL);
	struct tm ts;
	char clocalerror[255] = "none";
	char clocalTime[80] = "";
	char cDefaultName[23] = "_results.ainetwork.csv";
	std::string tmpCurrentFormula = "";
	std::string tmpCurrentValue = "";
	
	fileResultingNetwork.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try {
		localtime_s(&ts, &t);
		strftime(clocalTime, 80, "%Y-%m-%d_%H.%M.%S", &ts);
		cResultingNetworkFileName = clocalTime + cResultingNetworkFileName;
		printf("saving data of %s to file %s ...\n", this->strInternalName.c_str(), cResultingNetworkFileName.c_str());
		fileResultingNetwork.open(cResultingNetworkFileName.c_str(), std::fstream::out | std::fstream::app);
		
		fileResultingNetwork << "output network " << this->strInternalName << " with formula\n";

		// TODO THIS PART DOES NOT WORK
		printf("get maxi layer %4i ", this->getNumberOfLayers());

		std::string strFileContents = "";
		for (size_t iNode = 1; iNode <= this->getMaximumNodesLayer(true); iNode++)
		{
			// begin network and calculation output
			strFileContents = "";

			for (unsigned int iLayer = 0; iLayer <= this->getNumberOfLayers(); iLayer++)
			{
				unsigned int iNumNodesLayer = 0;
				iNumNodesLayer = this->getNumberOfNodesInLayer(max(1, iLayer));
				if (iNode > iNumNodesLayer)
				{
					break;
				}
				else
				{
					if (iLayer == 0)
					{
						// write name column
						strFileContents = this->TrainingDataColumnName((unsigned int)iNode) + ",";
					}
					else if (iLayer == 1)
					{
						strFileContents = strFileContents + std::to_string(this->vecValues[iNode]) + ",";
					}
					else
					{
						unsigned int iNumLayerStart = this->getLayerStart(iLayer);
						if (iNode <= iNumLayerStart + this->getNumberOfNodesInLayer(iLayer))
						{
							// clear value an set it to theshold
							tmpCurrentValue = std::to_string(-1 * this->vecThresholds[iNumLayerStart - 1 + iNode]);
							//create formula for all nodes in previous layer
							unsigned int iNumNodesPrevLayer = this->getNumberOfNodesInLayer(iLayer - 1);
							unsigned int iNumPrevLayerStart = this->getLayerStart(iLayer - 1);
							for (unsigned int x = 1; x <= iNumNodesPrevLayer; x++)
							{
								tmpCurrentValue += "-" + this->getExcelColumn(iLayer) + std::to_string(x+1);// times 1 is due to first line content
								tmpCurrentValue += "*" + std::to_string(this->vecWeights[iNumPrevLayerStart -1+x][iNumLayerStart - 1 + iNode]);// times weight
							}

							strFileContents += this->NodeFunctionXLS((unsigned int)iNode,tmpCurrentValue);
							strFileContents += ",";		// end of cell
						}
					}
				}
			}
			this->generateFileOutput(strFileContents);
			fileResultingNetwork << strFileContents << "\n"; 	// write line to file
		}

		fileResultingNetwork << "--- Weight as result from " << cResultingNetworkFileName << " on " << clocalTime << "---\n";
		
		std::string tmpFileContents = "node,thresholds,";
		for (unsigned int i = 1; i < iMaxNodes; i++)
		{
			tmpFileContents = tmpFileContents + std::to_string(i) + ","; //enumerating x-axis
		}
		tmpFileContents = tmpFileContents + "\n"; // end of first line
		for (unsigned int y = 1; y < this->vecWeights.size(); y++)
		{
			tmpFileContents = tmpFileContents + "node" + std::to_string(y) + " to x," + std::to_string(this->vecThresholds[y]) + ","; //output first column to remind users of position of matrix
			for (unsigned int x = 1; x < this->vecWeights[y].size(); x++)
			{
				tmpFileContents = tmpFileContents + std::to_string(this->vecWeights[y][x]) + ",";//output the real data
			}
			tmpFileContents = tmpFileContents + "\n";// end of row;
		}
		tmpFileContents = tmpFileContents + "--- Weights end ---\n";
		this->generateFileOutput(tmpFileContents);
		fileResultingNetwork << tmpFileContents;

		// write the training data to file
		tmpFileContents = "-- training data begin -- \nRow,";
		for (unsigned int iRow = 1; iRow < vvTrainingDataMatrix.at(0).size(); iRow++)
		{
			tmpFileContents = tmpFileContents + this->TrainingDataColumnName(iRow) + ",";
		}
		for (unsigned int iRow = 0; iRow<vvErrors.at(0).size(); iRow++)
		{
			tmpFileContents = tmpFileContents + "Error_" + std::to_string(iRow+1) + ",";
		}
		tmpFileContents = tmpFileContents + "\n";
		for (unsigned int iLine = 0; iLine < vvTrainingDataMatrix.size(); iLine++)
		{
			tmpFileContents = tmpFileContents + std::to_string(iLine)+",";
			// write training data
			for(unsigned int iRow=1; iRow<vvTrainingDataMatrix.at(iLine).size(); iRow++)
			{
				tmpFileContents = tmpFileContents + std::to_string( vvTrainingDataMatrix.at(iLine).at(iRow) )+ ",";
			}
			// write errors
			for (unsigned int iRow = 0; iRow<vvErrors.at(iLine).size(); iRow++)
			{
				tmpFileContents = tmpFileContents + std::to_string(vvErrors.at(iLine).at(iRow)) + ",";
			}
			tmpFileContents = tmpFileContents + "\n";
		}
		tmpFileContents = tmpFileContents + "-- training data end -- \n";
		this->generateFileOutput(tmpFileContents);
		fileResultingNetwork << tmpFileContents;

		fileResultingNetwork.close();
		printf("Data saved to %s\n", cResultingNetworkFileName.c_str());
	}
	catch (...) {
		fileResultingNetwork.close();
		strerror_s(clocalerror, errno);
		printf("ERROR while saving data: %s.\n", clocalerror);
	}
}

void AINetClass::setActivationFunction(unsigned int typeOfActivationFunction, unsigned int specificLayer)
{
	// this function is used to set the activation function of all layer or for one specific layer (if 2nd parameter is set)
	if (specificLayer > 0)
	{
		// set activation function for specified layer
		this->viLayerActivationFunction.at(min(specificLayer-1, viLayerActivationFunction.size() - 1)) = typeOfActivationFunction;
	}
	else
	{
		this->iActivationFunction = typeOfActivationFunction;
		for (unsigned int i = 0; i < this->viLayerActivationFunction.size(); i++)
		{
			this->viLayerActivationFunction.at(i) = typeOfActivationFunction;
		}
	}
}

void AINetClass::setDataFileName(std::string strFileName)
{
	// set the file name for data input
	this->theAIDataFileName = strFileName;
}

void AINetClass::setInternalName(std::string strIntName)
{
	// set the internal name of the class
	this->strInternalName = strIntName;
}

void AINetClass::setOptionAutoGenerate(bool bAutoGenerate)
{
	// set option for automatic generation of internal network
	this->bOptionAutoGenerate = bAutoGenerate;
}

void AINetClass::connectNodes(bool bFullyConnected, unsigned int iRandSeed)
{
	// this connects all nodes in the neural network
	this->bHasBeenConnected = true;
	// first do the auto-generation if parameter is set
	this->autoGenerateInternalNetwork();
	// TODO allow smart connected network by removing next line of code
	bFullyConnected = true;
	//variables
	unsigned int tmpTotalNumberNodes = 0;

	// seeding random number generator
	srand(iRandSeed);

	//function
	if (initializationDone)
	{
		tmpTotalNumberNodes = this->NUMNODES();

		for (unsigned int x = 1; x <= tmpTotalNumberNodes; x++) {
			if (bFullyConnected)
			{
				for (unsigned int y = 1; y <= tmpTotalNumberNodes; y++) {
					// all connections are created, except connections to self
					if (x == y)
					{
						this->vecWeights[x][y] = 0.0;
					}
					else
					{
						// TODO: BUG RANDOM NUMBER is the same for every run of this funtion
						this->vecWeights[x][y] = (rand() % 200) / 100.0;
					}
				}
			}
			else
			{
				// only valid & used connections are created
				for (unsigned int y = this->NUMINPUTNODES() + 1; tmpTotalNumberNodes; y++)
				{
					// generate node connections for all valid weights
					// next line is correct, but if wont work becuse vdNetworkTopology does not represent the network w/ historic data
					if (this->getLayerByNode(x) == this->getLayerByNode(y) - 1)
					{
						// node y is element of layer after node x
						this->vecWeights[x][y] = (rand() % 200 / 100.0);
					}
				}
			}
		}
		// generating thresholds for all nodes except input nodes
		for (unsigned int i = this->NUMINPUTNODES() + 1; i <= tmpTotalNumberNodes; i++)
		{
			this->vecThresholds[i] = rand() / (double)rand();
		}
	}
	else
	{
		this->throwFailure("network not properly initialized",true);
	}
}

void AINetClass::closeTrainingDataFile(std::ifstream &ptrDataFile)
{
	// closes training data file
	if (ptrDataFile.is_open())
	{
		// good, file is open.
		ptrDataFile.close();
	}
	else
	{
		printf("File couldn't be closed.\n");
	}
}

void AINetClass::combineNetworks(AINetClass& ptrAiNetClass, std::mutex & ptrMutex, unsigned int iNumber)
{
	// combining networks
	std::lock_guard<std::mutex> guard(ptrMutex);
	printf("\n Error this:\t%8f.4", this->calculateErrorMSE(-1));
	printf("\n Error ptr:\t%8f.4", ptrAiNetClass.calculateErrorMSE(-1));
	//TODO:BUG Check if sorting works
	ptrAiNetClass.sortNetwork();

	// TODO:DEBUG saving is just for debug
	ptrAiNetClass.saveResultingNetwork(iNumber);
	
	// combine date according to combination mode
	if (this->iThreadedCombinationMode == 0)
	{
		// combine all data
		// todo write data combination code.
		// todo this is not ready yet
	}

	Sleep(1000);
}

void AINetClass::createNetwork(std::vector<std::string> tmpsNetwork)
{
	// create network based on std::vector<std::string> first empty or zero line will end creation
	std::vector<unsigned int> tmpviNetwork;
	tmpviNetwork.resize(tmpsNetwork.size(), 0);
	for (unsigned int i = 0; i < tmpviNetwork.size(); i++)
	{
		tmpviNetwork.at(i) = atoi(tmpsNetwork.at(i).c_str());
	}
	this->createNetwork(tmpviNetwork);
}

void AINetClass::createNetwork(std::vector<unsigned int> tmpviNetwork)
{
	// create network based on std::vector<unsigned int> first line with zero will end creation
	unsigned int iNumLayers = 2;
	for (unsigned int i = 0; i < tmpviNetwork.size(); i++)
	{
		if (tmpviNetwork.at(i) > 0)
		{
			// set new maximum
			iNumLayers = max(iNumLayers, i);
			// continue
		}
		else
		{
			// this is zero so there cannot be any network.
			break;
		}
	}
	this->setNumberOfHiddenLayers(max(1, iNumLayers - 1));
	this->setNumInputNodes(tmpviNetwork.at(0));
	this->setNumOutputNodes(tmpviNetwork.at(iNumLayers));
	for (unsigned int i = 1; i < iNumLayers; i++)
	{
		// set the number of nodes in each hidden layer
		this->setNumberOfNodesinLayer(i+1, tmpviNetwork.at(i));
	}
}

void AINetClass::trainLine()
{
	// copy input node data from matrix to input notes
	// training with data

	// setting input nodes

	unsigned int tmpCurrentRow = this->CurrentTrainingDataRow();

	for (unsigned int i = 0; i <= this->iNumRealInputNodes; i++)
	{
		this->vecValues[i] = this->vvTrainingDataMatrix.at(tmpCurrentRow).at(i) + this->dInputOffset;
	}

	// now adding historic data for the input
	for (unsigned int h = 1; h <= (unsigned int)abs(this->iTimePreviousRows); h++)
	{
		if (this->iTimeNumInputColumns == 0)
		{
			// for each previous row add all input node values
			for (unsigned int i = 1; i <= this->iNumRealInputNodes; i++)
			{
				this->vecValues[(h*this->iNumRealInputNodes+i)] = this->getTrainingDataValue(tmpCurrentRow + h, i) + this->dInputOffset;
			}
		}
		else
		{
			// for each row select the first (iTimeNumInputColumns) columns.
			for (unsigned int i = 1; i <= this->iTimeNumInputColumns; i++)
			{
				this->vecValues[this->iNumRealInputNodes+((h-1)*this->iTimeNumInputColumns) + i] = this->getTrainingDataValue(tmpCurrentRow + h, i) + this->dInputOffset;
			}
		}
		// TODO add handling for previous/historic data
	}

	// setting output nodes
	for (unsigned int i = 1; i <= this->getNumberOfOutputNodes(); i++)
	{
		this->vecExpectedValues[this->getLayerStart(-1) -1 + i] = this->getTrainingDataValue(tmpCurrentRow,this->iNumRealInputNodes+i);
	}
}

void AINetClass::trainNetwork(bool bSilent)
{
	// training the network after all initialization is done
	if (bSilent)
	{
		this->bOptionDisplayAllNodes = false;
		this->bOptionStatus = false;
	}
	while (this->continueCalculation())
	{
			// start trainging with data loaded from file
			this->trainLine();

		this->activateNetwork();

		double sumOfSquaredErrors = 0.0;

		sumOfSquaredErrors = this->updateWeights();

		// calculate the Worst error and output
		if (this->IsTrainingRestart())
		{
			printf("Max Error at Row %i with value %8.6f\n", iWorstErrorRow, dWorstError);
			iWorstErrorRow = 0;
			dWorstError = 0.0;
		}
		else if (max(dWorstError, sumOfSquaredErrors) == sumOfSquaredErrors)
		{
			iWorstErrorRow = this->Counter() % this->getTrainingDataRowsMax();
			dWorstError = max(dWorstError, sumOfSquaredErrors);
		}
		this->displayIO(sumOfSquaredErrors); // if options set displaying all net

		if (this->bOptionDisplayAllNodes)
		{
			this->displayAllNodes(sumOfSquaredErrors); // if options ist set displaying all nodes
		}
		this->Counter(true);
	}
}

void AINetClass::displayIO(double sumOfSquaredErrors)
{
	if (this->bOptionIO)
	{
		if (this->IsTrainingRestart())
		{
			printf("\nNew Row    |");
			for (unsigned int i = 1; i <= this->NUMREALINPUTNODES() + this->NUMOUTPUTNODES(); i++)
			{
				if (this->TrainingDataColumns() >= i)
				{
					printf("%s|", this->TrainingDataColumnName(i).c_str());
				}
			}
			printf("\n");
		}
		printf("%8i:Row: %8i|", this->Counter(false), this->CurrentTrainingDataRow());
		for (unsigned int i = this->getLayerStart(1); i <= this->getLayerStart(1,false); i++)
		{
			//listing all input nodes
			if (i <= this->NUMREALINPUTNODES())
			{
				printf("%4.4f|", this->vecValues[i]);
			}
			else
			{
				printf("%4.4f!", this->vecValues[i]);
			}
		}
		for (unsigned int i = this->getLayerStart(-1); i <= this->getLayerStart(-1,false); i++)
		{
			//listing all outputnodes nodes
			printf("%8.4f", this->vecValues[i]);
			printf("(%8.4f)|", this->vecExpectedValues[i]);
		}
		printf("err:%8.5f", sumOfSquaredErrors);
		// if historic data is used create a new line for each of them
		printf("\n");
	}
	else
	{
		if (this->IsTrainingRestart())
		{
			printf("\n");
		}
		if (sumOfSquaredErrors > 1)
		{
			printf("|");
		}
		else if (sumOfSquaredErrors > 0.1)
		{
			printf("l");
		}
		else if (sumOfSquaredErrors > 0.01)
		{
			printf("i");
		}
		else if(sumOfSquaredErrors > 0.001)
		{
			printf(",");
		}
		else
		{
			printf(".");
		}
	}
}

void AINetClass::displayWeights()
{
	unsigned int iMaxNodes = 0;
	iMaxNodes = (unsigned int) this->vecWeights.size();
	// TODO: optionsWeight not mplemented yet
	if (this->optionWeight)
	{
		printf("--- Weights ---\n");
		// output Weight to screen
		printf("node;thresholds;");
		for (unsigned int i = 1; i <= iMaxNodes; i++)
		{
			printf("%i;", i);//enumerating x-axis
		}
		printf("\n"); // end of first line
		for (unsigned int y = 1; y < this->vecWeights.size(); y++)
		{
			printf("from node %i to x;", y); //output first column to remind users of position of matrix
			printf("%8.4f;", this->vecThresholds[y]);
			for (unsigned int x = 1; x < this->vecWeights[y].size(); x++)
			{
				printf("%8.4f;", this->vecWeights[y][x]); //output the real data
			}
			printf("\n");// end of row;
		}
		printf("--- Weights end ---\n");
	}
}

void AINetClass::displayStatus()
{
	// display status message
	if (this->bOptionStatus)
	{
		printf("\n--- STATUS ---\nNUMINPUTNODES=\t%i(%i)\n", this->NUMREALINPUTNODES(), this->NUMINPUTNODES());
		printf("NUMOUTPUTNODES=\t%i\n", this->NUMOUTPUTNODES());
		printf("NUMNODES=\t%i\n", this->NUMNODES());
		printf("MAXITERATIONS=\t%i, of which %i have been performed\n", this->getMaxIterations(), this->Counter()); //131072;
		printf("LEARNINGRATE=\t%8.4f\n", this->LearningRate());
		printf("training lines = \t%8i\n", this->getTrainingDataRowsMax());
		printf("-- Options --\n");
		printf("SHUFFLE:\t%s\n", this->bOptionShuffle ? "true" : "false");
		printf("AUTO-GENERATE:\t%s\n",this->bOptionAutoGenerate ? "true" : "false");
		printf("CSV-GER:\t%s\n", this->bOptionCSVGER ? "true" : "false");
		printf("passes=\t%8i\t with %8i additional rows\n", this->getMaxIterations() / this->getTrainingDataRowsMax(), this->getMaxIterations() % this->getTrainingDataRowsMax());
		if (this->iTimeNextRows == 0 && this->iTimePreviousRows == 0)
		{
			printf("TIME_DEPENCY:\tOFF\n");
		}
		else
		{
			printf("TIME_DEPENCY:\tON\n\tin\t%4i rows with first %4i nodes\n\tout\t%4i rows with first %4i nodes", this->iTimePreviousRows, this->iTimeNumInputColumns, this->iTimeNextRows, this->iTimeNumOutputColumns);
		}
		printf("\nneural network with %8i layers", this->getNumberOfLayers());
		for (unsigned int numLayers = 1; numLayers <= this->getNumberOfLayers(); numLayers++)
		{
			printf("\nlayer %4i with %8i nodes, from node %4i to %4i", numLayers, this->getNumberOfNodesInLayer(numLayers), this->getLayerStart(numLayers), this->getLayerStart(numLayers, false));
			if (numLayers == 1)printf(" (input) ");
			if (numLayers == this->getNumberOfLayers())printf(" (output) ");
		}
		printf("--- STATUS ---\n");
	}
}

void AINetClass::displayAllNodes(double sumOfSquaredErrors)
{
	// display all nodes
	if (this->IsTrainingRestart())
		printf("Display Whole Network and Error---------------------\n");
	
	// get maximum number of nodes (maximum rows)
	for (unsigned int i = 1; i <= this->getMaximumNodesLayer(true); i++)
	{
		// get maximum number of layers (max cols)
		for (unsigned int j = 1; j <= this->getNumberOfLayers();  j++)
		{
			// do the output magic
			if (i <= this->getNumberOfNodesInLayer(j))
			{
				printf("%2i:%8.4f|", (this->getLayerStart(j) + i - 1), this->getNodeValue(this->getLayerStart(j) + i -1));
			}
			else
				printf("--:---.----|");
		}
		printf("\n");
	}
	printf("\n------ err: %8.5f\n", sumOfSquaredErrors);
}

std::string AINetClass::getDataFileName()
{
	if (this->theAIDataFileName == "")
		return "default.csv";
	else
		return this->theAIDataFileName;
}

std::string AINetClass::getExcelColumn(size_t stColumn)
{
	// this converts a number of columns to excel alphabetic numbering
	std::string abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	std::string returnString = "";
	if (max(1,stColumn) <= 26)
	{
		// only on letter is returned
		returnString = abc.at(stColumn - 1);
	}
	else if (stColumn <= 676)
	{
		// 676 is 26*26 aka ZZ ;-)
		returnString = abc.at((int)(stColumn / 26)); // yes it cuts of the end
		returnString = returnString + abc.at(stColumn % 26);
	}
	return returnString;
}

double AINetClass::updateWeights()
{
	double sumOfSquaredErrors = 0.0;
	// OLD:
	// sumOfSquaredErrors = aincNetwork.updateWeights();
	// NEW:
	for (int iLayer = this->getNumberOfLayers(); iLayer >=1; iLayer--)
	{
		sumOfSquaredErrors += this->updateWeightsInLayer(iLayer);
	}
	return sumOfSquaredErrors;
}

double AINetClass::getNodeValue(unsigned int tmpNode)
{
	// returns value of node
	unsigned int iReturn;
	iReturn = (unsigned int) min(max(0, tmpNode),this->vecValues.size()-1);
	if (iReturn==tmpNode)
	{
		return this->vecValues.at(iReturn);
	}
	else
	{
		return 0.0;
	}
}

std::vector<std::string> AINetClass::getErrorList()
{
	// this one returns the error list
	return this->errorList;
}

std::vector<std::string> AINetClass::splitString(const std::string & strInput, const std::string & strDelimiter)
{
	// spliting String
	std::vector<std::string> strElements;

	for (size_t stStart = 0, stEnd; stStart < strInput.length(); stStart = stEnd + strDelimiter.length())
	{
		size_t stPosition = strInput.find(strDelimiter, stStart);
		stEnd = stPosition != std::string::npos ? stPosition : strInput.length();

		std::string strElement = strInput.substr(stStart, stEnd - stStart);

		strElements.push_back(strElement);

	}

	if (strInput.empty() || (strInput.substr(strInput.size()-strDelimiter.size(), strDelimiter.size()) == strDelimiter))
	{
		strElements.push_back("");
	}

	return strElements;
}

bool AINetClass::IsNetworkReady()
{
	//this one should test if network is ready to run.
	return this->initializationDone;
}

bool AINetClass::autoGenerateInternalNetwork()
{
	// generating internal network 
	unsigned int iAutoInput = this->NUMINPUTNODES();
	unsigned int iAutoOutput = this->NUMOUTPUTNODES();
	unsigned int iAutoLayer = (unsigned int)this->vdNetworkTopology.capacity();
	double iAutoResult = 1;
	if (this->bOptionAutoGenerate)
	{
		// option auto generate is set
		for (unsigned int i = 2; i < iAutoLayer; i++)
		{
			iAutoResult = max(1,min(iAutoInput, round(iAutoInput - 0.8 * ((iAutoInput - iAutoOutput) / iAutoLayer * i))));
			this->setNumberOfNodesinLayer(i, (unsigned int)iAutoResult);
		}
		this->resizeVectors();
		return true;
	}
	else
	{
		// return false if network is not automaticaly generated
		return false;
	}
}

double AINetClass::getTrainingDataValue(unsigned int row, unsigned int column)
{
	// safeAccesstoTrainingData
	double tmpReturn=0;
	if (row < this->vvTrainingDataMatrix.size())
	{
		if (column < this->vvTrainingDataMatrix[row].size())
		{
			tmpReturn= this->vvTrainingDataMatrix[row][column];
		}
		else
		{
			this->throwFailure("column exeeded", true);
		}
	}
	else
	{
		this->throwFailure("row exeeded", true);
	}
	return tmpReturn;
}

double AINetClass::updateWeightsInLayer(int tmpLayer)
{
	// updates the weight in specific layer
	this->validLayer(tmpLayer);
	unsigned int tmpLayerBegin = this->getLayerStart(tmpLayer);
	unsigned int tmpLayerEnd = this->getLayerStart(tmpLayer, false);
	
	double sumOfSquaredError = 0.0;
	unsigned int n = 0;

	for (unsigned int iCurrentNode = tmpLayerBegin; iCurrentNode <= tmpLayerEnd; iCurrentNode++)
	{
		double dErrorAtCurrentNode = 0.0;
		// calculation of delta
		if (tmpLayer <= 1)
		{
			this->bBackpropagationActive = false;
		}
		else if(this->IsLastLayer(tmpLayer))
		{
			this->bBackpropagationActive = true;
			// calculate absolute error from output
			dErrorAtCurrentNode = this->vecValues[iCurrentNode] - this->vecExpectedValues[iCurrentNode];
			this->vvErrors.at(this->CurrentTrainingDataRow()).at(iCurrentNode - tmpLayerBegin) = dErrorAtCurrentNode;
			sumOfSquaredError += pow(dErrorAtCurrentNode, 2);
			// calculation of delta at current node
			double deltaAtNode = this->NodeFunction(this->vecValues[iCurrentNode], iCurrentNode, true) * dErrorAtCurrentNode;
			// saving delta at current node
			this->vecCalcDelta[iCurrentNode] = deltaAtNode;
		}
		else
		{
			this->bBackpropagationActive = true;
			// calculate error from following layer
			for (unsigned int iLaterNode = this->getLayerStart(tmpLayer + 1); iLaterNode <= this->getLayerStart(tmpLayer + 1, false); iLaterNode++)
			{
				// correcting the weights at first
				dErrorAtCurrentNode += this->vecCalcDelta[iLaterNode] * this->vecWeights[iCurrentNode][iLaterNode];
			}
			double deltaAtNode = this->NodeFunction(this->vecValues[iCurrentNode], iCurrentNode, true) * dErrorAtCurrentNode;
			this->vecCalcDelta[iCurrentNode] = deltaAtNode;
		}
		// calculaton of weights from all nodes in previous layer to iCurrentNode
		if (tmpLayer <= 1)
		{
			// there are no weights to input nodes.
		}
		else
		{
			for (unsigned int iPreviousNode = this->getLayerStart(tmpLayer - 1); iPreviousNode <= this->getLayerStart(tmpLayer - 1, false); iPreviousNode++)
			{
				double deltaWeight = -1 * this->LearningRate() * this->vecCalcDelta[iCurrentNode] * this->vecValues[iPreviousNode];
				this->vecWeights.at(iPreviousNode).at(iCurrentNode) += deltaWeight;
				// adjusting thresholds
				this->vecThresholds[iCurrentNode] += this->LearningRate() * this->vecCalcDelta[iCurrentNode];
			}
		}
		n = iCurrentNode;
	}
	return(sumOfSquaredError / max(n, 1));
}

inline void AINetClass::ReplaceAllStrings(std::string & str, const std::string & from, const std::string & to)
{
	size_t	start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); //Handles case where 'to' is  a substring of 'from'
	}
	//return str;
}

unsigned int AINetClass::getNumberOfInputNodes()
{
	if (this->vdNetworkTopology.size() > 0)
	{
		return this->vdNetworkTopology.at(0);
	}
	else
	{
		return this->iNumInputNodes;
	}
}

unsigned int AINetClass::getNumberOfOutputNodes()
{
	if (this->vdNetworkTopology.size() > 0)
	{
		return this->vdNetworkTopology.at(this->vdNetworkTopology.size()-1);
	}
	else
	{
		return this->iNumOutputNodes;
	}
}

std::string AINetClass::generateFileOutput(std::string& strFileContents)
{
	// convert data to suitable format
	if (this->bOptionCSVGER)
	{
		ReplaceAllStrings(strFileContents, ",", ";");
		ReplaceAllStrings(strFileContents, ".", ",");
	}
	return strFileContents;
}

std::string AINetClass::generateFileInput(std::string & strFileContents)
{
	// convert data to suitable format
	if (this->bOptionCSVGER)
	{
		ReplaceAllStrings(strFileContents, ",", ".");
		ReplaceAllStrings(strFileContents, ";", ",");
	}
	return strFileContents;
}

bool AINetClass::recalculateInputDataPullList()
{
	if (this->iTrainingDataRowsMax > 0)
	{
		this->inputDataPullList.clear();
		this->inputDataPullList.resize(this->iTrainingDataRowsMax - abs(iTimePreviousRows));
		std::fill(this->inputDataPullList.begin(), this->inputDataPullList.end(), 0);
		for (unsigned int i = 0; i < this->inputDataPullList.capacity(); i++)
		{
			if (iTimePreviousRows < 0)
				this->inputDataPullList.at(i) = i - iTimePreviousRows; // add (2x negative) the offset at the beginning
			else
			{
				this->inputDataPullList.at(i) = i;	// simple generation of list
			}
		}
	}
	return (this->iTrainingDataRowsMax > 0);
}

bool AINetClass::throwFailure(std::string tmpError,bool doexit)
{
	// there is a failure in the network. 
	// i have to exit
	this->errorList.push_back(tmpError);
	if (doexit)
		exit(1);
	return true;
}

double AINetClass::calculateErrorMSE(int iLayer)
{
	// this function calculates the mean square error for specified layer
	this->validLayer(iLayer);
	double sumOfError = 0.0;
	double n = 0.0;
	for (unsigned int iCurrentNode = this->getLayerStart(iLayer); iCurrentNode <= this->getLayerStart(iLayer,false); iCurrentNode++)
	{
		sumOfError = pow(this->vecCalcDelta.at(iCurrentNode),2);
		n += 1;
	}
	if (n == 0.0)
		this->throwFailure("division by zero (n) in MSE Error calculation", true);
	return (sumOfError/n);
}

unsigned int AINetClass::resizeVectors()
{
	// recalculates the size of input vector()
	unsigned int tmpInputVectorSize = 0;
	if (this->iTimeNumInputColumns == 0)
	{
		tmpInputVectorSize += this->iNumRealInputNodes * (1 + abs(this->iTimePreviousRows));
	}
	else
	{
		tmpInputVectorSize += this->iNumRealInputNodes + this->iTimeNumInputColumns * abs(this->iTimePreviousRows);
	}
	this->vdNetworkTopology.at(0) = tmpInputVectorSize;
	for (unsigned int i = 1; i < vdNetworkTopology.size(); i++)
	{
		tmpInputVectorSize += vdNetworkTopology.at(i);
	}
	
	this->initialize();

	return tmpInputVectorSize;
}

int AINetClass::validLayer(int & tmpLayer, bool tmpRemoveOffset)
{
	// returning a valid layer between 1 and vdNetworkTopology.size()
	if (tmpLayer < 0)
	{
		// first of all, make it positive.
		// if tmpLayer is negative, it is counted from the last layer in the network
		// due to the fact that it has to be -1 or smaler, this->vdNetworkTopology.size() is always substractet at least 1 element which is added so -1 refers to last layer.
		tmpLayer = max(1, (int) this->vdNetworkTopology.size() + 1 + tmpLayer);
	}
	else if (tmpLayer > 0)
	{
		// great it is greater than 0, so the first (aka input layer ist 1)
		tmpLayer = min((int) this->vdNetworkTopology.size(), tmpLayer);
	}
	else
	{
		// refering to first layer
		tmpLayer = 1;
	}
	// next part will remove the offset from the layer
	if (tmpRemoveOffset)
	{
		// it may not be below zero!
		tmpLayer = max(0, tmpLayer - 1);
	}
	return tmpLayer;
}

double AINetClass::NodeFunction(double weightedInput, unsigned int currentNodeID, bool derivative)
{
	// this function changes the activation function of the nodes
	double dActivationFunctionResult = 0.0;

	switch (this->getActivationFunction(currentNodeID))
	{
	case 1:
		// using tanh
		if (derivative) { dActivationFunctionResult = 1 / pow(cosh(-weightedInput), 2); } 
		else { dActivationFunctionResult = tanh(weightedInput); }
		break;
	case 2:
		//BIP
		if (derivative) { dActivationFunctionResult = 2 * pow(E, weightedInput) / pow((pow(E, weightedInput) + 1), 2); }
		else { dActivationFunctionResult = (1 - pow(E, -weightedInput)) / (1 + pow(E, -weightedInput)); }
		break;
	case 3:
		// using rectified linear unit
		if (derivative) { dActivationFunctionResult = 1; }
		else { dActivationFunctionResult = max(0, weightedInput); }
		break;
	case 4:
		// 
		if (derivative) { dActivationFunctionResult = pow(E, -weightedInput) / pow((pow(E, -weightedInput) + 1), 2); }
		else { dActivationFunctionResult = 1.0 / (1.0 + pow(E, -weightedInput)); }
		break;
	default:
		if (derivative) { dActivationFunctionResult = pow(E, -weightedInput) / pow((pow(E, -weightedInput) + 1), 2); }
		else { dActivationFunctionResult = 1.0 / (1.0 + pow(E, -weightedInput)); }
		break;
	}
	return dActivationFunctionResult;
}

std::string AINetClass::NodeFunctionXLS(unsigned int tmpNode, std::string tmpCalculatedInput)
{
	// calculate xls output
	std::string myActivationFunction = "";
	unsigned int tmpActivationFunction = this->getActivationFunction(tmpNode);
	switch (tmpActivationFunction)
	{
	case 1:
		// using tanh
		myActivationFunction = "=TANHYP(%d)";
		break;
	case 2:
		//BIP
		myActivationFunction = "=(1-EXP(-%d))/(1+EXP(-%d))";
		break;
	case 3:
		//reLu
		myActivationFunction = "=MAX(0,%d)";
		break;
	case 4:
		// using linear activation function on output layer
		myActivationFunction = "=1.0/(1.0+EXP(-%d))";
		break;
	default:
		myActivationFunction = "=1.0/(1.0+EXP(-%d))";
		break;
	}

	ReplaceAllStrings(myActivationFunction, "%d", tmpCalculatedInput);
	//ReplaceAllStrings(myActivationFunction, "--", "");

	return myActivationFunction;
}

std::string AINetClass::NodeFunctionJS(unsigned int tmpNode, std::string tmpCalculatedInput)
{
	// TODO WRITE CODE for NodeFunctionJS
	return std::string();
}
