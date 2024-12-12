/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import * as dataset from "./dataset";

/** 存储控件是否隐藏时添加的状态后缀 */
const HIDE_STATE_SUFFIX = "_hide";

/** 名称和激活函数之间的映射 */
export let activations: {[key: string]: nn.ActivationFunction} = {
  "relu": nn.Activations.RELU,
  "tanh": nn.Activations.TANH,
  "sigmoid": nn.Activations.SIGMOID,
  "linear": nn.Activations.LINEAR,
  "leakyRelu": nn.Activations.LEAKY_RELU
};

/** 名称和正则化函数之间的映射 */
export let regularizations: {[key: string]: nn.RegularizationFunction} = {
  "none": null,
  "L1": nn.RegularizationFunction.L1,
  "L2": nn.RegularizationFunction.L2
};

/** 数据集名称和生成分类数据的函数之间的映射 */
export let datasets: {[key: string]: dataset.DataGenerator} = {
  "circle": dataset.classifyCircleData,
  "xor": dataset.classifyXORData,
  "gauss": dataset.classifyTwoGaussData,
  "spiral": dataset.classifySpiralData,
};

/** 数据集名称和生成回归数据的函数之间的映射 */
export let regDatasets: {[key: string]: dataset.DataGenerator} = {
  "reg-plane": dataset.regressPlane,
  "reg-gauss": dataset.regressGaussian
};

export function getKeyFromValue(obj: any, value: any): string {
  for (let key in obj) {
    if (obj[key] === value) {
      return key;
    }
  }
  return undefined;
}

function endsWith(s: string, suffix: string): boolean {
  return s.substr(-suffix.length) === suffix;
}

function getHideProps(obj: any): string[] {
  let result: string[] = [];
  for (let prop in obj) {
    if (endsWith(prop, HIDE_STATE_SUFFIX)) {
      result.push(prop);
    }
  }
  return result;
}

/**
 * 状态变量的数据类型。用于确定序列化和反序列化方法。
 */
export enum Type {
  STRING,
  NUMBER,
  ARRAY_NUMBER,
  ARRAY_STRING,
  BOOLEAN,
  OBJECT
}

export enum Problem {
  CLASSIFICATION,
  REGRESSION
}

export let problems = {
  "classification": Problem.CLASSIFICATION,
  "regression": Problem.REGRESSION
};

export interface Property {
  name: string;
  type: Type;
  keyMap?: {[key: string]: any};
};

// 添加GUI状态
export class State {

  private static PROPS: Property[] = [
    {name: "activation", type: Type.OBJECT, keyMap: activations},
    {name: "regularization", type: Type.OBJECT, keyMap: regularizations},
    {name: "batchSize", type: Type.NUMBER},
    {name: "dataset", type: Type.OBJECT, keyMap: datasets},
    {name: "regDataset", type: Type.OBJECT, keyMap: regDatasets},
    {name: "learningRate", type: Type.NUMBER},
    {name: "regularizationRate", type: Type.NUMBER},
    {name: "noise", type: Type.NUMBER},
    {name: "networkShape", type: Type.ARRAY_NUMBER},
    {name: "seed", type: Type.STRING},
    {name: "showTestData", type: Type.BOOLEAN},
    {name: "discretize", type: Type.BOOLEAN},
    {name: "percTrainData", type: Type.NUMBER},
    {name: "x", type: Type.BOOLEAN},
    {name: "y", type: Type.BOOLEAN},
    {name: "xTimesY", type: Type.BOOLEAN},
    {name: "xSquared", type: Type.BOOLEAN},
    {name: "ySquared", type: Type.BOOLEAN},
    {name: "cosX", type: Type.BOOLEAN},
    {name: "sinX", type: Type.BOOLEAN},
    {name: "cosY", type: Type.BOOLEAN},
    {name: "sinY", type: Type.BOOLEAN},
    {name: "collectStats", type: Type.BOOLEAN},
    {name: "tutorial", type: Type.STRING},
    {name: "problem", type: Type.OBJECT, keyMap: problems},
    {name: "initZero", type: Type.BOOLEAN},
    {name: "hideText", type: Type.BOOLEAN}
  ];

  [key: string]: any;
  learningRate = 0.03;
  regularizationRate = 0;
  showTestData = false;
  noise = 0;
  batchSize = 10;
  discretize = false;
  tutorial: string = null;
  percTrainData = 50;
  activation = nn.Activations.TANH;
  regularization: nn.RegularizationFunction = null;
  problem = Problem.CLASSIFICATION;
  initZero = false;
  hideText = false;
  collectStats = false;
  numHiddenLayers = 1;
  hiddenLayerControls: any[] = [];
  networkShape: number[] = [4, 2];
  x = true;
  y = true;
  xTimesY = false;
  xSquared = false;
  ySquared = false;
  cosX = false;
  sinX = false;
  cosY = false;
  sinY = false;
  dataset: dataset.DataGenerator = dataset.classifyCircleData;
  regDataset: dataset.DataGenerator = dataset.regressPlane;
  seed: string;

  /**
   * 从URL哈希中反序列化状态
   */
  static deserializeState(): State {
    let map: {[key: string]: string} = {};
    for (let keyvalue of window.location.hash.slice(1).split("&")) {
      let [name, value] = keyvalue.split("=");
      map[name] = value;
    }
    let state = new State();

    function hasKey(name: string): boolean {
      return name in map && map[name] != null && map[name].trim() !== "";
    }

    function parseArray(value: string): string[] {
      return value.trim() === "" ? [] : value.split(",");
    }

    // 反序列化常规属性
    State.PROPS.forEach(({name, type, keyMap}) => {
      switch (type) {
        case Type.OBJECT:
          if (keyMap == null) {
            throw Error("A key-value map must be provided for state " +
                "variables of type Object");
          }
          if (hasKey(name) && map[name] in keyMap) {
            state[name] = keyMap[map[name]];
          }
          break;
        case Type.NUMBER:
          if (hasKey(name)) {
            // + 运算符用于将字符串转换为数字
            state[name] = +map[name];
          }
          break;
        case Type.STRING:
          if (hasKey(name)) {
            state[name] = map[name];
          }
          break;
        case Type.BOOLEAN:
          if (hasKey(name)) {
            state[name] = (map[name] === "false" ? false : true);
          }
          break;
        case Type.ARRAY_NUMBER:
          if (name in map) {
            state[name] = parseArray(map[name]).map(Number);
          }
          break;
        case Type.ARRAY_STRING:
          if (name in map) {
            state[name] = parseArray(map[name]);
          }
          break;
        default:
          throw Error("Encountered an unknown type for a state variable");
      }
    });

    // 反序列化对应于隐藏UI控件的状态属性
    getHideProps(map).forEach(prop => {
      state[prop] = (map[prop] === "true") ? true : false;
    });
    state.numHiddenLayers = state.networkShape.length;
    if (state.seed == null) {
      state.seed = Math.random().toFixed(5);
    }
    Math.seedrandom(state.seed);
    return state;
  }

  /**
   * 将状态序列化到URL哈希中
   */
  serialize() {
    // 序列化常规属性
    let props: string[] = [];
    State.PROPS.forEach(({name, type, keyMap}) => {
      let value = this[name];
      // 不序列化缺失的值
      if (value == null) {
        return;
      }
      if (type === Type.OBJECT) {
        value = getKeyFromValue(keyMap, value);
      } else if (type === Type.ARRAY_NUMBER ||
          type === Type.ARRAY_STRING) {
        value = value.join(",");
      }
      props.push(`${name}=${value}`);
    });
    // 序列化对应于隐藏UI控件的属性
    getHideProps(this).forEach(prop => {
      props.push(`${prop}=${this[prop]}`);
    });
    window.location.hash = props.join("&");
  }

  /** 返回所有隐藏的属性 */
  getHiddenProps(): string[] {
    let result: string[] = [];
    for (let prop in this) {
      if (endsWith(prop, HIDE_STATE_SUFFIX) && String(this[prop]) === "true") {
        result.push(prop.replace(HIDE_STATE_SUFFIX, ""));
      }
    }
    return result;
  }

  setHideProperty(name: string, hidden: boolean) {
    this[name + HIDE_STATE_SUFFIX] = hidden;
  }
}
