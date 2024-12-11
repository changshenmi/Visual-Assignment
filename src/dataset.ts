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

import * as d3 from 'd3';

/**
 * Example2D 类型表示一个二维的例子，包含 x 和 y 坐标以及标签。
 */
export type Example2D = {
  x: number,
  y: number,
  label: number
};

type Point = {
  x: number,
  y: number
};

/**
 * 使用 Fisher-Yates 算法对数组进行洗牌。
 * 使用 seedrandom 库作为随机数生成器
 */
export function shuffle(array: any[]): void {
  let counter = array.length;
  let temp = 0;
  let index = 0;
  // 当数组中有数据时
  while (counter > 0) {
    // 随机选择一个索引
    index = Math.floor(Math.random() * counter);
    // 交换的过程是从最后一个数据到第一个数据，减1向前一个位置
    counter--;
    // 交换当前元素和选中的元素
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

export type DataGenerator = (numSamples: number, noise: number) => Example2D[];

/**
 * 生成两个正态分布的样本
 * @param numSamples ：样本数量
 * @param noise ：噪声
 * @returns ：Example2D类型的数组
 */
export function classifyTwoGaussData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];//points是Example2D类型的数组

  let varianceScale = d3.scale.linear().domain([0, .5]).range([0.5, 4]); //将0到0.5的输入等比例映射到0.5到4的输出
  let variance = varianceScale(noise); //将noise映射到0.5到4的输出  

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  }

  genGauss(2, 2, 1); // 正态分布正样本
  genGauss(-2, -2, -1); // 正态分布负样本
  return points;
}

/**
 * 生成平面样本
 * @param numSamples ：样本数量
 * @param noise ：噪声
 * @returns ：Example2D类型的数组
 */ 
export function regressPlane(numSamples: number, noise: number):
  Example2D[] {
  let radius = 6;
  let labelScale = d3.scale.linear()
    .domain([-10, 10])
    .range([-1, 1]);//将-10到10的输入等比例映射到-1到1的输出
  let getLabel = (x, y) => labelScale(x + y);//将x+y映射到-1到1的输出

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);//x 在-6到6之间 
    let y = randUniform(-radius, radius);//y 在-6到6之间
    let noiseX = randUniform(-radius, radius) * noise;//乘以noise
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);//将x+noiseX和y+noiseY映射到-1到1的输出
    points.push({x, y, label});//将x,y,label添加到points数组中
  }
  return points;
}

/**
 * 生成高斯样本
 * @param numSamples ：样本数量
 * @param noise ：噪声
 * @returns ：Example2D类型的数组
 */ 
export function regressGaussian(numSamples: number, noise: number):
  Example2D[] {
  let points: Example2D[] = [];

  let labelScale = d3.scale.linear()
    .domain([0, 2])
    .range([1, 0])
    .clamp(true);

  let gaussians = [
    [-4, 2.5, 1],
    [0, 2.5, -1],
    [4, 2.5, 1],
    [-4, -2.5, -1],
    [0, -2.5, 1],
    [4, -2.5, -1]
  ];

  function getLabel(x, y) {
    // Choose the one that is maximum in abs value.
    let label = 0;
    gaussians.forEach(([cx, cy, sign]) => {
      let newLabel = sign * labelScale(dist({x, y}, {x: cx, y: cy}));
      if (Math.abs(newLabel) > Math.abs(label)) {
        label = newLabel;
      }
    });
    return label;
  }
  let radius = 6;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({x, y, label});
  };
  return points;
}

export function classifySpiralData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = i / n * 5;
      let t = 1.75 * i / n * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  }

  genSpiral(0, 1); // Positive examples.
  genSpiral(Math.PI, -1); // Negative examples.
  return points;
}

export function classifyCircleData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];
  let radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }//如果p到center的距离小于radius的一半，则label为1，否则为-1
  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(radius * 0.7, radius); //并不是靠近radius * 0.5，而是从radius * 0.7开始
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }
  return points;
}

export function classifyXORData(numSamples: number, noise: number):
    Example2D[] {
  function getXORLabel(p: Point) { return p.x * p.y >= 0 ? 1 : -1; }

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5);
    let padding = 0.3;
    x += x > 0 ? padding : -padding;  // Padding.
    let y = randUniform(-5, 5);
    y += y > 0 ? padding : -padding;
    let noiseX = randUniform(-5, 5) * noise;
    let noiseY = randUniform(-5, 5) * noise;
    let label = getXORLabel({x: x + noiseX, y: y + noiseY});
    points.push({x, y, label});
  }
  return points;
}

/**
 * 返回一个均匀分布的随机数
 * 使用seedrandom库作为随机数生成器
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * 从正态分布中采样。使用seedrandom库作为随机数生成器
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean = 0, variance = 1): number {
  let v1: number, v2: number, s: number;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

/** 返回两个点之间的欧几里得距离 */ 
function dist(a: Point, b: Point): number {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}
