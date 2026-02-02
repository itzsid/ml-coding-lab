var my=Object.defineProperty;var hy=(e,t,n)=>t in e?my(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n;var $t=(e,t,n)=>hy(e,typeof t!="symbol"?t+"":t,n);function gy(e,t){for(var n=0;n<t.length;n++){const r=t[n];if(typeof r!="string"&&!Array.isArray(r)){for(const i in r)if(i!=="default"&&!(i in e)){const a=Object.getOwnPropertyDescriptor(r,i);a&&Object.defineProperty(e,i,a.get?a:{enumerable:!0,get:()=>r[i]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const a of i)if(a.type==="childList")for(const o of a.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&r(o)}).observe(document,{childList:!0,subtree:!0});function n(i){const a={};return i.integrity&&(a.integrity=i.integrity),i.referrerPolicy&&(a.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?a.credentials="include":i.crossOrigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function r(i){if(i.ep)return;i.ep=!0;const a=n(i);fetch(i.href,a)}})();var Xa=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function Zn(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}var uf={exports:{}},To={},cf={exports:{}},J={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Xi=Symbol.for("react.element"),yy=Symbol.for("react.portal"),xy=Symbol.for("react.fragment"),vy=Symbol.for("react.strict_mode"),_y=Symbol.for("react.profiler"),wy=Symbol.for("react.provider"),by=Symbol.for("react.context"),ky=Symbol.for("react.forward_ref"),Sy=Symbol.for("react.suspense"),Cy=Symbol.for("react.memo"),Ey=Symbol.for("react.lazy"),jc=Symbol.iterator;function zy(e){return e===null||typeof e!="object"?null:(e=jc&&e[jc]||e["@@iterator"],typeof e=="function"?e:null)}var df={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},pf=Object.assign,ff={};function qr(e,t,n){this.props=e,this.context=t,this.refs=ff,this.updater=n||df}qr.prototype.isReactComponent={};qr.prototype.setState=function(e,t){if(typeof e!="object"&&typeof e!="function"&&e!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,e,t,"setState")};qr.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,"forceUpdate")};function mf(){}mf.prototype=qr.prototype;function hu(e,t,n){this.props=e,this.context=t,this.refs=ff,this.updater=n||df}var gu=hu.prototype=new mf;gu.constructor=hu;pf(gu,qr.prototype);gu.isPureReactComponent=!0;var Rc=Array.isArray,hf=Object.prototype.hasOwnProperty,yu={current:null},gf={key:!0,ref:!0,__self:!0,__source:!0};function yf(e,t,n){var r,i={},a=null,o=null;if(t!=null)for(r in t.ref!==void 0&&(o=t.ref),t.key!==void 0&&(a=""+t.key),t)hf.call(t,r)&&!gf.hasOwnProperty(r)&&(i[r]=t[r]);var s=arguments.length-2;if(s===1)i.children=n;else if(1<s){for(var l=Array(s),u=0;u<s;u++)l[u]=arguments[u+2];i.children=l}if(e&&e.defaultProps)for(r in s=e.defaultProps,s)i[r]===void 0&&(i[r]=s[r]);return{$$typeof:Xi,type:e,key:a,ref:o,props:i,_owner:yu.current}}function Ty(e,t){return{$$typeof:Xi,type:e.type,key:t,ref:e.ref,props:e.props,_owner:e._owner}}function xu(e){return typeof e=="object"&&e!==null&&e.$$typeof===Xi}function Py(e){var t={"=":"=0",":":"=2"};return"$"+e.replace(/[=:]/g,function(n){return t[n]})}var Ic=/\/+/g;function ts(e,t){return typeof e=="object"&&e!==null&&e.key!=null?Py(""+e.key):t.toString(36)}function ja(e,t,n,r,i){var a=typeof e;(a==="undefined"||a==="boolean")&&(e=null);var o=!1;if(e===null)o=!0;else switch(a){case"string":case"number":o=!0;break;case"object":switch(e.$$typeof){case Xi:case yy:o=!0}}if(o)return o=e,i=i(o),e=r===""?"."+ts(o,0):r,Rc(i)?(n="",e!=null&&(n=e.replace(Ic,"$&/")+"/"),ja(i,t,n,"",function(u){return u})):i!=null&&(xu(i)&&(i=Ty(i,n+(!i.key||o&&o.key===i.key?"":(""+i.key).replace(Ic,"$&/")+"/")+e)),t.push(i)),1;if(o=0,r=r===""?".":r+":",Rc(e))for(var s=0;s<e.length;s++){a=e[s];var l=r+ts(a,s);o+=ja(a,t,n,l,i)}else if(l=zy(e),typeof l=="function")for(e=l.call(e),s=0;!(a=e.next()).done;)a=a.value,l=r+ts(a,s++),o+=ja(a,t,n,l,i);else if(a==="object")throw t=String(e),Error("Objects are not valid as a React child (found: "+(t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)+"). If you meant to render a collection of children, use an array instead.");return o}function sa(e,t,n){if(e==null)return e;var r=[],i=0;return ja(e,r,"","",function(a){return t.call(n,a,i++)}),r}function Ny(e){if(e._status===-1){var t=e._result;t=t(),t.then(function(n){(e._status===0||e._status===-1)&&(e._status=1,e._result=n)},function(n){(e._status===0||e._status===-1)&&(e._status=2,e._result=n)}),e._status===-1&&(e._status=0,e._result=t)}if(e._status===1)return e._result.default;throw e._result}var $e={current:null},Ra={transition:null},Ay={ReactCurrentDispatcher:$e,ReactCurrentBatchConfig:Ra,ReactCurrentOwner:yu};function xf(){throw Error("act(...) is not supported in production builds of React.")}J.Children={map:sa,forEach:function(e,t,n){sa(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return sa(e,function(){t++}),t},toArray:function(e){return sa(e,function(t){return t})||[]},only:function(e){if(!xu(e))throw Error("React.Children.only expected to receive a single React element child.");return e}};J.Component=qr;J.Fragment=xy;J.Profiler=_y;J.PureComponent=hu;J.StrictMode=vy;J.Suspense=Sy;J.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=Ay;J.act=xf;J.cloneElement=function(e,t,n){if(e==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+e+".");var r=pf({},e.props),i=e.key,a=e.ref,o=e._owner;if(t!=null){if(t.ref!==void 0&&(a=t.ref,o=yu.current),t.key!==void 0&&(i=""+t.key),e.type&&e.type.defaultProps)var s=e.type.defaultProps;for(l in t)hf.call(t,l)&&!gf.hasOwnProperty(l)&&(r[l]=t[l]===void 0&&s!==void 0?s[l]:t[l])}var l=arguments.length-2;if(l===1)r.children=n;else if(1<l){s=Array(l);for(var u=0;u<l;u++)s[u]=arguments[u+2];r.children=s}return{$$typeof:Xi,type:e.type,key:i,ref:a,props:r,_owner:o}};J.createContext=function(e){return e={$$typeof:by,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},e.Provider={$$typeof:wy,_context:e},e.Consumer=e};J.createElement=yf;J.createFactory=function(e){var t=yf.bind(null,e);return t.type=e,t};J.createRef=function(){return{current:null}};J.forwardRef=function(e){return{$$typeof:ky,render:e}};J.isValidElement=xu;J.lazy=function(e){return{$$typeof:Ey,_payload:{_status:-1,_result:e},_init:Ny}};J.memo=function(e,t){return{$$typeof:Cy,type:e,compare:t===void 0?null:t}};J.startTransition=function(e){var t=Ra.transition;Ra.transition={};try{e()}finally{Ra.transition=t}};J.unstable_act=xf;J.useCallback=function(e,t){return $e.current.useCallback(e,t)};J.useContext=function(e){return $e.current.useContext(e)};J.useDebugValue=function(){};J.useDeferredValue=function(e){return $e.current.useDeferredValue(e)};J.useEffect=function(e,t){return $e.current.useEffect(e,t)};J.useId=function(){return $e.current.useId()};J.useImperativeHandle=function(e,t,n){return $e.current.useImperativeHandle(e,t,n)};J.useInsertionEffect=function(e,t){return $e.current.useInsertionEffect(e,t)};J.useLayoutEffect=function(e,t){return $e.current.useLayoutEffect(e,t)};J.useMemo=function(e,t){return $e.current.useMemo(e,t)};J.useReducer=function(e,t,n){return $e.current.useReducer(e,t,n)};J.useRef=function(e){return $e.current.useRef(e)};J.useState=function(e){return $e.current.useState(e)};J.useSyncExternalStore=function(e,t,n){return $e.current.useSyncExternalStore(e,t,n)};J.useTransition=function(){return $e.current.useTransition()};J.version="18.3.1";cf.exports=J;var E=cf.exports;const Ne=Zn(E),Ly=gy({__proto__:null,default:Ne},[E]);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var jy=E,Ry=Symbol.for("react.element"),Iy=Symbol.for("react.fragment"),Oy=Object.prototype.hasOwnProperty,Dy=jy.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,My={key:!0,ref:!0,__self:!0,__source:!0};function vf(e,t,n){var r,i={},a=null,o=null;n!==void 0&&(a=""+n),t.key!==void 0&&(a=""+t.key),t.ref!==void 0&&(o=t.ref);for(r in t)Oy.call(t,r)&&!My.hasOwnProperty(r)&&(i[r]=t[r]);if(e&&e.defaultProps)for(r in t=e.defaultProps,t)i[r]===void 0&&(i[r]=t[r]);return{$$typeof:Ry,type:e,key:a,ref:o,props:i,_owner:Dy.current}}To.Fragment=Iy;To.jsx=vf;To.jsxs=vf;uf.exports=To;var h=uf.exports,el={},_f={exports:{}},ut={},wf={exports:{}},bf={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(e){function t(O,V){var _=O.length;O.push(V);e:for(;0<_;){var B=_-1>>>1,Q=O[B];if(0<i(Q,V))O[B]=V,O[_]=Q,_=B;else break e}}function n(O){return O.length===0?null:O[0]}function r(O){if(O.length===0)return null;var V=O[0],_=O.pop();if(_!==V){O[0]=_;e:for(var B=0,Q=O.length,w=Q>>>1;B<w;){var ie=2*(B+1)-1,be=O[ie],le=ie+1,dt=O[le];if(0>i(be,_))le<Q&&0>i(dt,be)?(O[B]=dt,O[le]=_,B=le):(O[B]=be,O[ie]=_,B=ie);else if(le<Q&&0>i(dt,_))O[B]=dt,O[le]=_,B=le;else break e}}return V}function i(O,V){var _=O.sortIndex-V.sortIndex;return _!==0?_:O.id-V.id}if(typeof performance=="object"&&typeof performance.now=="function"){var a=performance;e.unstable_now=function(){return a.now()}}else{var o=Date,s=o.now();e.unstable_now=function(){return o.now()-s}}var l=[],u=[],d=1,c=null,p=3,f=!1,y=!1,v=!1,k=typeof setTimeout=="function"?setTimeout:null,m=typeof clearTimeout=="function"?clearTimeout:null,g=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function x(O){for(var V=n(u);V!==null;){if(V.callback===null)r(u);else if(V.startTime<=O)r(u),V.sortIndex=V.expirationTime,t(l,V);else break;V=n(u)}}function C(O){if(v=!1,x(O),!y)if(n(l)!==null)y=!0,$(z);else{var V=n(u);V!==null&&ne(C,V.startTime-O)}}function z(O,V){y=!1,v&&(v=!1,m(N),N=-1),f=!0;var _=p;try{for(x(V),c=n(l);c!==null&&(!(c.expirationTime>V)||O&&!I());){var B=c.callback;if(typeof B=="function"){c.callback=null,p=c.priorityLevel;var Q=B(c.expirationTime<=V);V=e.unstable_now(),typeof Q=="function"?c.callback=Q:c===n(l)&&r(l),x(V)}else r(l);c=n(l)}if(c!==null)var w=!0;else{var ie=n(u);ie!==null&&ne(C,ie.startTime-V),w=!1}return w}finally{c=null,p=_,f=!1}}var S=!1,T=null,N=-1,F=5,b=-1;function I(){return!(e.unstable_now()-b<F)}function W(){if(T!==null){var O=e.unstable_now();b=O;var V=!0;try{V=T(!0,O)}finally{V?X():(S=!1,T=null)}}else S=!1}var X;if(typeof g=="function")X=function(){g(W)};else if(typeof MessageChannel<"u"){var ee=new MessageChannel,A=ee.port2;ee.port1.onmessage=W,X=function(){A.postMessage(null)}}else X=function(){k(W,0)};function $(O){T=O,S||(S=!0,X())}function ne(O,V){N=k(function(){O(e.unstable_now())},V)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(O){O.callback=null},e.unstable_continueExecution=function(){y||f||(y=!0,$(z))},e.unstable_forceFrameRate=function(O){0>O||125<O?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):F=0<O?Math.floor(1e3/O):5},e.unstable_getCurrentPriorityLevel=function(){return p},e.unstable_getFirstCallbackNode=function(){return n(l)},e.unstable_next=function(O){switch(p){case 1:case 2:case 3:var V=3;break;default:V=p}var _=p;p=V;try{return O()}finally{p=_}},e.unstable_pauseExecution=function(){},e.unstable_requestPaint=function(){},e.unstable_runWithPriority=function(O,V){switch(O){case 1:case 2:case 3:case 4:case 5:break;default:O=3}var _=p;p=O;try{return V()}finally{p=_}},e.unstable_scheduleCallback=function(O,V,_){var B=e.unstable_now();switch(typeof _=="object"&&_!==null?(_=_.delay,_=typeof _=="number"&&0<_?B+_:B):_=B,O){case 1:var Q=-1;break;case 2:Q=250;break;case 5:Q=1073741823;break;case 4:Q=1e4;break;default:Q=5e3}return Q=_+Q,O={id:d++,callback:V,priorityLevel:O,startTime:_,expirationTime:Q,sortIndex:-1},_>B?(O.sortIndex=_,t(u,O),n(l)===null&&O===n(u)&&(v?(m(N),N=-1):v=!0,ne(C,_-B))):(O.sortIndex=Q,t(l,O),y||f||(y=!0,$(z))),O},e.unstable_shouldYield=I,e.unstable_wrapCallback=function(O){var V=p;return function(){var _=p;p=V;try{return O.apply(this,arguments)}finally{p=_}}}})(bf);wf.exports=bf;var Fy=wf.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var qy=E,lt=Fy;function j(e){for(var t="https://reactjs.org/docs/error-decoder.html?invariant="+e,n=1;n<arguments.length;n++)t+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+e+"; visit "+t+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var kf=new Set,Ei={};function Jn(e,t){Nr(e,t),Nr(e+"Capture",t)}function Nr(e,t){for(Ei[e]=t,e=0;e<t.length;e++)kf.add(t[e])}var Jt=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),tl=Object.prototype.hasOwnProperty,Wy=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,Oc={},Dc={};function By(e){return tl.call(Dc,e)?!0:tl.call(Oc,e)?!1:Wy.test(e)?Dc[e]=!0:(Oc[e]=!0,!1)}function Uy(e,t,n,r){if(n!==null&&n.type===0)return!1;switch(typeof t){case"function":case"symbol":return!0;case"boolean":return r?!1:n!==null?!n.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function Vy(e,t,n,r){if(t===null||typeof t>"u"||Uy(e,t,n,r))return!0;if(r)return!1;if(n!==null)switch(n.type){case 3:return!t;case 4:return t===!1;case 5:return isNaN(t);case 6:return isNaN(t)||1>t}return!1}function Qe(e,t,n,r,i,a,o){this.acceptsBooleans=t===2||t===3||t===4,this.attributeName=r,this.attributeNamespace=i,this.mustUseProperty=n,this.propertyName=e,this.type=t,this.sanitizeURL=a,this.removeEmptyString=o}var Re={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){Re[e]=new Qe(e,0,!1,e,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var t=e[0];Re[t]=new Qe(t,1,!1,e[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(e){Re[e]=new Qe(e,2,!1,e.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){Re[e]=new Qe(e,2,!1,e,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){Re[e]=new Qe(e,3,!1,e.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(e){Re[e]=new Qe(e,3,!0,e,null,!1,!1)});["capture","download"].forEach(function(e){Re[e]=new Qe(e,4,!1,e,null,!1,!1)});["cols","rows","size","span"].forEach(function(e){Re[e]=new Qe(e,6,!1,e,null,!1,!1)});["rowSpan","start"].forEach(function(e){Re[e]=new Qe(e,5,!1,e.toLowerCase(),null,!1,!1)});var vu=/[\-:]([a-z])/g;function _u(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var t=e.replace(vu,_u);Re[t]=new Qe(t,1,!1,e,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var t=e.replace(vu,_u);Re[t]=new Qe(t,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(e){var t=e.replace(vu,_u);Re[t]=new Qe(t,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(e){Re[e]=new Qe(e,1,!1,e.toLowerCase(),null,!1,!1)});Re.xlinkHref=new Qe("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(e){Re[e]=new Qe(e,1,!1,e.toLowerCase(),null,!0,!0)});function wu(e,t,n,r){var i=Re.hasOwnProperty(t)?Re[t]:null;(i!==null?i.type!==0:r||!(2<t.length)||t[0]!=="o"&&t[0]!=="O"||t[1]!=="n"&&t[1]!=="N")&&(Vy(t,n,i,r)&&(n=null),r||i===null?By(t)&&(n===null?e.removeAttribute(t):e.setAttribute(t,""+n)):i.mustUseProperty?e[i.propertyName]=n===null?i.type===3?!1:"":n:(t=i.attributeName,r=i.attributeNamespace,n===null?e.removeAttribute(t):(i=i.type,n=i===3||i===4&&n===!0?"":""+n,r?e.setAttributeNS(r,t,n):e.setAttribute(t,n))))}var rn=qy.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,la=Symbol.for("react.element"),sr=Symbol.for("react.portal"),lr=Symbol.for("react.fragment"),bu=Symbol.for("react.strict_mode"),nl=Symbol.for("react.profiler"),Sf=Symbol.for("react.provider"),Cf=Symbol.for("react.context"),ku=Symbol.for("react.forward_ref"),rl=Symbol.for("react.suspense"),il=Symbol.for("react.suspense_list"),Su=Symbol.for("react.memo"),ln=Symbol.for("react.lazy"),Ef=Symbol.for("react.offscreen"),Mc=Symbol.iterator;function Kr(e){return e===null||typeof e!="object"?null:(e=Mc&&e[Mc]||e["@@iterator"],typeof e=="function"?e:null)}var xe=Object.assign,ns;function si(e){if(ns===void 0)try{throw Error()}catch(n){var t=n.stack.trim().match(/\n( *(at )?)/);ns=t&&t[1]||""}return`
`+ns+e}var rs=!1;function is(e,t){if(!e||rs)return"";rs=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(t)if(t=function(){throw Error()},Object.defineProperty(t.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(t,[])}catch(u){var r=u}Reflect.construct(e,[],t)}else{try{t.call()}catch(u){r=u}e.call(t.prototype)}else{try{throw Error()}catch(u){r=u}e()}}catch(u){if(u&&r&&typeof u.stack=="string"){for(var i=u.stack.split(`
`),a=r.stack.split(`
`),o=i.length-1,s=a.length-1;1<=o&&0<=s&&i[o]!==a[s];)s--;for(;1<=o&&0<=s;o--,s--)if(i[o]!==a[s]){if(o!==1||s!==1)do if(o--,s--,0>s||i[o]!==a[s]){var l=`
`+i[o].replace(" at new "," at ");return e.displayName&&l.includes("<anonymous>")&&(l=l.replace("<anonymous>",e.displayName)),l}while(1<=o&&0<=s);break}}}finally{rs=!1,Error.prepareStackTrace=n}return(e=e?e.displayName||e.name:"")?si(e):""}function Hy(e){switch(e.tag){case 5:return si(e.type);case 16:return si("Lazy");case 13:return si("Suspense");case 19:return si("SuspenseList");case 0:case 2:case 15:return e=is(e.type,!1),e;case 11:return e=is(e.type.render,!1),e;case 1:return e=is(e.type,!0),e;default:return""}}function al(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case lr:return"Fragment";case sr:return"Portal";case nl:return"Profiler";case bu:return"StrictMode";case rl:return"Suspense";case il:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case Cf:return(e.displayName||"Context")+".Consumer";case Sf:return(e._context.displayName||"Context")+".Provider";case ku:var t=e.render;return e=e.displayName,e||(e=t.displayName||t.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case Su:return t=e.displayName||null,t!==null?t:al(e.type)||"Memo";case ln:t=e._payload,e=e._init;try{return al(e(t))}catch{}}return null}function $y(e){var t=e.type;switch(e.tag){case 24:return"Cache";case 9:return(t.displayName||"Context")+".Consumer";case 10:return(t._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=t.render,e=e.displayName||e.name||"",t.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return t;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return al(t);case 8:return t===bu?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t}return null}function Cn(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function zf(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(t==="checkbox"||t==="radio")}function Qy(e){var t=zf(e)?"checked":"value",n=Object.getOwnPropertyDescriptor(e.constructor.prototype,t),r=""+e[t];if(!e.hasOwnProperty(t)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var i=n.get,a=n.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(o){r=""+o,a.call(this,o)}}),Object.defineProperty(e,t,{enumerable:n.enumerable}),{getValue:function(){return r},setValue:function(o){r=""+o},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function ua(e){e._valueTracker||(e._valueTracker=Qy(e))}function Tf(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r="";return e&&(r=zf(e)?e.checked?"true":"false":e.value),e=r,e!==n?(t.setValue(e),!0):!1}function Ya(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function ol(e,t){var n=t.checked;return xe({},t,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??e._wrapperState.initialChecked})}function Fc(e,t){var n=t.defaultValue==null?"":t.defaultValue,r=t.checked!=null?t.checked:t.defaultChecked;n=Cn(t.value!=null?t.value:n),e._wrapperState={initialChecked:r,initialValue:n,controlled:t.type==="checkbox"||t.type==="radio"?t.checked!=null:t.value!=null}}function Pf(e,t){t=t.checked,t!=null&&wu(e,"checked",t,!1)}function sl(e,t){Pf(e,t);var n=Cn(t.value),r=t.type;if(n!=null)r==="number"?(n===0&&e.value===""||e.value!=n)&&(e.value=""+n):e.value!==""+n&&(e.value=""+n);else if(r==="submit"||r==="reset"){e.removeAttribute("value");return}t.hasOwnProperty("value")?ll(e,t.type,n):t.hasOwnProperty("defaultValue")&&ll(e,t.type,Cn(t.defaultValue)),t.checked==null&&t.defaultChecked!=null&&(e.defaultChecked=!!t.defaultChecked)}function qc(e,t,n){if(t.hasOwnProperty("value")||t.hasOwnProperty("defaultValue")){var r=t.type;if(!(r!=="submit"&&r!=="reset"||t.value!==void 0&&t.value!==null))return;t=""+e._wrapperState.initialValue,n||t===e.value||(e.value=t),e.defaultValue=t}n=e.name,n!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,n!==""&&(e.name=n)}function ll(e,t,n){(t!=="number"||Ya(e.ownerDocument)!==e)&&(n==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+n&&(e.defaultValue=""+n))}var li=Array.isArray;function _r(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t["$"+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty("$"+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=""+Cn(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function ul(e,t){if(t.dangerouslySetInnerHTML!=null)throw Error(j(91));return xe({},t,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function Wc(e,t){var n=t.value;if(n==null){if(n=t.children,t=t.defaultValue,n!=null){if(t!=null)throw Error(j(92));if(li(n)){if(1<n.length)throw Error(j(93));n=n[0]}t=n}t==null&&(t=""),n=t}e._wrapperState={initialValue:Cn(n)}}function Nf(e,t){var n=Cn(t.value),r=Cn(t.defaultValue);n!=null&&(n=""+n,n!==e.value&&(e.value=n),t.defaultValue==null&&e.defaultValue!==n&&(e.defaultValue=n)),r!=null&&(e.defaultValue=""+r)}function Bc(e){var t=e.textContent;t===e._wrapperState.initialValue&&t!==""&&t!==null&&(e.value=t)}function Af(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function cl(e,t){return e==null||e==="http://www.w3.org/1999/xhtml"?Af(t):e==="http://www.w3.org/2000/svg"&&t==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var ca,Lf=function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(t,n,r,i){MSApp.execUnsafeLocalFunction(function(){return e(t,n,r,i)})}:e}(function(e,t){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=t;else{for(ca=ca||document.createElement("div"),ca.innerHTML="<svg>"+t.valueOf().toString()+"</svg>",t=ca.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;t.firstChild;)e.appendChild(t.firstChild)}});function zi(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var fi={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},Ky=["Webkit","ms","Moz","O"];Object.keys(fi).forEach(function(e){Ky.forEach(function(t){t=t+e.charAt(0).toUpperCase()+e.substring(1),fi[t]=fi[e]})});function jf(e,t,n){return t==null||typeof t=="boolean"||t===""?"":n||typeof t!="number"||t===0||fi.hasOwnProperty(e)&&fi[e]?(""+t).trim():t+"px"}function Rf(e,t){e=e.style;for(var n in t)if(t.hasOwnProperty(n)){var r=n.indexOf("--")===0,i=jf(n,t[n],r);n==="float"&&(n="cssFloat"),r?e.setProperty(n,i):e[n]=i}}var Xy=xe({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function dl(e,t){if(t){if(Xy[e]&&(t.children!=null||t.dangerouslySetInnerHTML!=null))throw Error(j(137,e));if(t.dangerouslySetInnerHTML!=null){if(t.children!=null)throw Error(j(60));if(typeof t.dangerouslySetInnerHTML!="object"||!("__html"in t.dangerouslySetInnerHTML))throw Error(j(61))}if(t.style!=null&&typeof t.style!="object")throw Error(j(62))}}function pl(e,t){if(e.indexOf("-")===-1)return typeof t.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var fl=null;function Cu(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var ml=null,wr=null,br=null;function Uc(e){if(e=Zi(e)){if(typeof ml!="function")throw Error(j(280));var t=e.stateNode;t&&(t=jo(t),ml(e.stateNode,e.type,t))}}function If(e){wr?br?br.push(e):br=[e]:wr=e}function Of(){if(wr){var e=wr,t=br;if(br=wr=null,Uc(e),t)for(e=0;e<t.length;e++)Uc(t[e])}}function Df(e,t){return e(t)}function Mf(){}var as=!1;function Ff(e,t,n){if(as)return e(t,n);as=!0;try{return Df(e,t,n)}finally{as=!1,(wr!==null||br!==null)&&(Mf(),Of())}}function Ti(e,t){var n=e.stateNode;if(n===null)return null;var r=jo(n);if(r===null)return null;n=r[t];e:switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(r=!r.disabled)||(e=e.type,r=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!r;break e;default:e=!1}if(e)return null;if(n&&typeof n!="function")throw Error(j(231,t,typeof n));return n}var hl=!1;if(Jt)try{var Xr={};Object.defineProperty(Xr,"passive",{get:function(){hl=!0}}),window.addEventListener("test",Xr,Xr),window.removeEventListener("test",Xr,Xr)}catch{hl=!1}function Yy(e,t,n,r,i,a,o,s,l){var u=Array.prototype.slice.call(arguments,3);try{t.apply(n,u)}catch(d){this.onError(d)}}var mi=!1,Ga=null,Za=!1,gl=null,Gy={onError:function(e){mi=!0,Ga=e}};function Zy(e,t,n,r,i,a,o,s,l){mi=!1,Ga=null,Yy.apply(Gy,arguments)}function Jy(e,t,n,r,i,a,o,s,l){if(Zy.apply(this,arguments),mi){if(mi){var u=Ga;mi=!1,Ga=null}else throw Error(j(198));Za||(Za=!0,gl=u)}}function er(e){var t=e,n=e;if(e.alternate)for(;t.return;)t=t.return;else{e=t;do t=e,t.flags&4098&&(n=t.return),e=t.return;while(e)}return t.tag===3?n:null}function qf(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function Vc(e){if(er(e)!==e)throw Error(j(188))}function ex(e){var t=e.alternate;if(!t){if(t=er(e),t===null)throw Error(j(188));return t!==e?null:e}for(var n=e,r=t;;){var i=n.return;if(i===null)break;var a=i.alternate;if(a===null){if(r=i.return,r!==null){n=r;continue}break}if(i.child===a.child){for(a=i.child;a;){if(a===n)return Vc(i),e;if(a===r)return Vc(i),t;a=a.sibling}throw Error(j(188))}if(n.return!==r.return)n=i,r=a;else{for(var o=!1,s=i.child;s;){if(s===n){o=!0,n=i,r=a;break}if(s===r){o=!0,r=i,n=a;break}s=s.sibling}if(!o){for(s=a.child;s;){if(s===n){o=!0,n=a,r=i;break}if(s===r){o=!0,r=a,n=i;break}s=s.sibling}if(!o)throw Error(j(189))}}if(n.alternate!==r)throw Error(j(190))}if(n.tag!==3)throw Error(j(188));return n.stateNode.current===n?e:t}function Wf(e){return e=ex(e),e!==null?Bf(e):null}function Bf(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var t=Bf(e);if(t!==null)return t;e=e.sibling}return null}var Uf=lt.unstable_scheduleCallback,Hc=lt.unstable_cancelCallback,tx=lt.unstable_shouldYield,nx=lt.unstable_requestPaint,ke=lt.unstable_now,rx=lt.unstable_getCurrentPriorityLevel,Eu=lt.unstable_ImmediatePriority,Vf=lt.unstable_UserBlockingPriority,Ja=lt.unstable_NormalPriority,ix=lt.unstable_LowPriority,Hf=lt.unstable_IdlePriority,Po=null,Wt=null;function ax(e){if(Wt&&typeof Wt.onCommitFiberRoot=="function")try{Wt.onCommitFiberRoot(Po,e,void 0,(e.current.flags&128)===128)}catch{}}var Lt=Math.clz32?Math.clz32:lx,ox=Math.log,sx=Math.LN2;function lx(e){return e>>>=0,e===0?32:31-(ox(e)/sx|0)|0}var da=64,pa=4194304;function ui(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function eo(e,t){var n=e.pendingLanes;if(n===0)return 0;var r=0,i=e.suspendedLanes,a=e.pingedLanes,o=n&268435455;if(o!==0){var s=o&~i;s!==0?r=ui(s):(a&=o,a!==0&&(r=ui(a)))}else o=n&~i,o!==0?r=ui(o):a!==0&&(r=ui(a));if(r===0)return 0;if(t!==0&&t!==r&&!(t&i)&&(i=r&-r,a=t&-t,i>=a||i===16&&(a&4194240)!==0))return t;if(r&4&&(r|=n&16),t=e.entangledLanes,t!==0)for(e=e.entanglements,t&=r;0<t;)n=31-Lt(t),i=1<<n,r|=e[n],t&=~i;return r}function ux(e,t){switch(e){case 1:case 2:case 4:return t+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function cx(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,a=e.pendingLanes;0<a;){var o=31-Lt(a),s=1<<o,l=i[o];l===-1?(!(s&n)||s&r)&&(i[o]=ux(s,t)):l<=t&&(e.expiredLanes|=s),a&=~s}}function yl(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function $f(){var e=da;return da<<=1,!(da&4194240)&&(da=64),e}function os(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function Yi(e,t,n){e.pendingLanes|=t,t!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,t=31-Lt(t),e[t]=n}function dx(e,t){var n=e.pendingLanes&~t;e.pendingLanes=t,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=t,e.mutableReadLanes&=t,e.entangledLanes&=t,t=e.entanglements;var r=e.eventTimes;for(e=e.expirationTimes;0<n;){var i=31-Lt(n),a=1<<i;t[i]=0,r[i]=-1,e[i]=-1,n&=~a}}function zu(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-Lt(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}var se=0;function Qf(e){return e&=-e,1<e?4<e?e&268435455?16:536870912:4:1}var Kf,Tu,Xf,Yf,Gf,xl=!1,fa=[],hn=null,gn=null,yn=null,Pi=new Map,Ni=new Map,cn=[],px="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function $c(e,t){switch(e){case"focusin":case"focusout":hn=null;break;case"dragenter":case"dragleave":gn=null;break;case"mouseover":case"mouseout":yn=null;break;case"pointerover":case"pointerout":Pi.delete(t.pointerId);break;case"gotpointercapture":case"lostpointercapture":Ni.delete(t.pointerId)}}function Yr(e,t,n,r,i,a){return e===null||e.nativeEvent!==a?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:a,targetContainers:[i]},t!==null&&(t=Zi(t),t!==null&&Tu(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function fx(e,t,n,r,i){switch(t){case"focusin":return hn=Yr(hn,e,t,n,r,i),!0;case"dragenter":return gn=Yr(gn,e,t,n,r,i),!0;case"mouseover":return yn=Yr(yn,e,t,n,r,i),!0;case"pointerover":var a=i.pointerId;return Pi.set(a,Yr(Pi.get(a)||null,e,t,n,r,i)),!0;case"gotpointercapture":return a=i.pointerId,Ni.set(a,Yr(Ni.get(a)||null,e,t,n,r,i)),!0}return!1}function Zf(e){var t=qn(e.target);if(t!==null){var n=er(t);if(n!==null){if(t=n.tag,t===13){if(t=qf(n),t!==null){e.blockedOn=t,Gf(e.priority,function(){Xf(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Ia(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=vl(e.domEventName,e.eventSystemFlags,t[0],e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);fl=r,n.target.dispatchEvent(r),fl=null}else return t=Zi(n),t!==null&&Tu(t),e.blockedOn=n,!1;t.shift()}return!0}function Qc(e,t,n){Ia(e)&&n.delete(t)}function mx(){xl=!1,hn!==null&&Ia(hn)&&(hn=null),gn!==null&&Ia(gn)&&(gn=null),yn!==null&&Ia(yn)&&(yn=null),Pi.forEach(Qc),Ni.forEach(Qc)}function Gr(e,t){e.blockedOn===t&&(e.blockedOn=null,xl||(xl=!0,lt.unstable_scheduleCallback(lt.unstable_NormalPriority,mx)))}function Ai(e){function t(i){return Gr(i,e)}if(0<fa.length){Gr(fa[0],e);for(var n=1;n<fa.length;n++){var r=fa[n];r.blockedOn===e&&(r.blockedOn=null)}}for(hn!==null&&Gr(hn,e),gn!==null&&Gr(gn,e),yn!==null&&Gr(yn,e),Pi.forEach(t),Ni.forEach(t),n=0;n<cn.length;n++)r=cn[n],r.blockedOn===e&&(r.blockedOn=null);for(;0<cn.length&&(n=cn[0],n.blockedOn===null);)Zf(n),n.blockedOn===null&&cn.shift()}var kr=rn.ReactCurrentBatchConfig,to=!0;function hx(e,t,n,r){var i=se,a=kr.transition;kr.transition=null;try{se=1,Pu(e,t,n,r)}finally{se=i,kr.transition=a}}function gx(e,t,n,r){var i=se,a=kr.transition;kr.transition=null;try{se=4,Pu(e,t,n,r)}finally{se=i,kr.transition=a}}function Pu(e,t,n,r){if(to){var i=vl(e,t,n,r);if(i===null)gs(e,t,r,no,n),$c(e,r);else if(fx(i,e,t,n,r))r.stopPropagation();else if($c(e,r),t&4&&-1<px.indexOf(e)){for(;i!==null;){var a=Zi(i);if(a!==null&&Kf(a),a=vl(e,t,n,r),a===null&&gs(e,t,r,no,n),a===i)break;i=a}i!==null&&r.stopPropagation()}else gs(e,t,r,null,n)}}var no=null;function vl(e,t,n,r){if(no=null,e=Cu(r),e=qn(e),e!==null)if(t=er(e),t===null)e=null;else if(n=t.tag,n===13){if(e=qf(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null);return no=e,null}function Jf(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(rx()){case Eu:return 1;case Vf:return 4;case Ja:case ix:return 16;case Hf:return 536870912;default:return 16}default:return 16}}var pn=null,Nu=null,Oa=null;function em(){if(Oa)return Oa;var e,t=Nu,n=t.length,r,i="value"in pn?pn.value:pn.textContent,a=i.length;for(e=0;e<n&&t[e]===i[e];e++);var o=n-e;for(r=1;r<=o&&t[n-r]===i[a-r];r++);return Oa=i.slice(e,1<r?1-r:void 0)}function Da(e){var t=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function ma(){return!0}function Kc(){return!1}function ct(e){function t(n,r,i,a,o){this._reactName=n,this._targetInst=i,this.type=r,this.nativeEvent=a,this.target=o,this.currentTarget=null;for(var s in e)e.hasOwnProperty(s)&&(n=e[s],this[s]=n?n(a):a[s]);return this.isDefaultPrevented=(a.defaultPrevented!=null?a.defaultPrevented:a.returnValue===!1)?ma:Kc,this.isPropagationStopped=Kc,this}return xe(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=ma)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=ma)},persist:function(){},isPersistent:ma}),t}var Wr={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Au=ct(Wr),Gi=xe({},Wr,{view:0,detail:0}),yx=ct(Gi),ss,ls,Zr,No=xe({},Gi,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:Lu,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==Zr&&(Zr&&e.type==="mousemove"?(ss=e.screenX-Zr.screenX,ls=e.screenY-Zr.screenY):ls=ss=0,Zr=e),ss)},movementY:function(e){return"movementY"in e?e.movementY:ls}}),Xc=ct(No),xx=xe({},No,{dataTransfer:0}),vx=ct(xx),_x=xe({},Gi,{relatedTarget:0}),us=ct(_x),wx=xe({},Wr,{animationName:0,elapsedTime:0,pseudoElement:0}),bx=ct(wx),kx=xe({},Wr,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),Sx=ct(kx),Cx=xe({},Wr,{data:0}),Yc=ct(Cx),Ex={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},zx={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},Tx={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function Px(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=Tx[e])?!!t[e]:!1}function Lu(){return Px}var Nx=xe({},Gi,{key:function(e){if(e.key){var t=Ex[e.key]||e.key;if(t!=="Unidentified")return t}return e.type==="keypress"?(e=Da(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?zx[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:Lu,charCode:function(e){return e.type==="keypress"?Da(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?Da(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),Ax=ct(Nx),Lx=xe({},No,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),Gc=ct(Lx),jx=xe({},Gi,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:Lu}),Rx=ct(jx),Ix=xe({},Wr,{propertyName:0,elapsedTime:0,pseudoElement:0}),Ox=ct(Ix),Dx=xe({},No,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),Mx=ct(Dx),Fx=[9,13,27,32],ju=Jt&&"CompositionEvent"in window,hi=null;Jt&&"documentMode"in document&&(hi=document.documentMode);var qx=Jt&&"TextEvent"in window&&!hi,tm=Jt&&(!ju||hi&&8<hi&&11>=hi),Zc=" ",Jc=!1;function nm(e,t){switch(e){case"keyup":return Fx.indexOf(t.keyCode)!==-1;case"keydown":return t.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function rm(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var ur=!1;function Wx(e,t){switch(e){case"compositionend":return rm(t);case"keypress":return t.which!==32?null:(Jc=!0,Zc);case"textInput":return e=t.data,e===Zc&&Jc?null:e;default:return null}}function Bx(e,t){if(ur)return e==="compositionend"||!ju&&nm(e,t)?(e=em(),Oa=Nu=pn=null,ur=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case"compositionend":return tm&&t.locale!=="ko"?null:t.data;default:return null}}var Ux={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function ed(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t==="input"?!!Ux[e.type]:t==="textarea"}function im(e,t,n,r){If(r),t=ro(t,"onChange"),0<t.length&&(n=new Au("onChange","change",null,n,r),e.push({event:n,listeners:t}))}var gi=null,Li=null;function Vx(e){hm(e,0)}function Ao(e){var t=pr(e);if(Tf(t))return e}function Hx(e,t){if(e==="change")return t}var am=!1;if(Jt){var cs;if(Jt){var ds="oninput"in document;if(!ds){var td=document.createElement("div");td.setAttribute("oninput","return;"),ds=typeof td.oninput=="function"}cs=ds}else cs=!1;am=cs&&(!document.documentMode||9<document.documentMode)}function nd(){gi&&(gi.detachEvent("onpropertychange",om),Li=gi=null)}function om(e){if(e.propertyName==="value"&&Ao(Li)){var t=[];im(t,Li,e,Cu(e)),Ff(Vx,t)}}function $x(e,t,n){e==="focusin"?(nd(),gi=t,Li=n,gi.attachEvent("onpropertychange",om)):e==="focusout"&&nd()}function Qx(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return Ao(Li)}function Kx(e,t){if(e==="click")return Ao(t)}function Xx(e,t){if(e==="input"||e==="change")return Ao(t)}function Yx(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var It=typeof Object.is=="function"?Object.is:Yx;function ji(e,t){if(It(e,t))return!0;if(typeof e!="object"||e===null||typeof t!="object"||t===null)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!tl.call(t,i)||!It(e[i],t[i]))return!1}return!0}function rd(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function id(e,t){var n=rd(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=rd(n)}}function sm(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?sm(e,t.parentNode):"contains"in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function lm(){for(var e=window,t=Ya();t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href=="string"}catch{n=!1}if(n)e=t.contentWindow;else break;t=Ya(e.document)}return t}function Ru(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||t==="textarea"||e.contentEditable==="true")}function Gx(e){var t=lm(),n=e.focusedElem,r=e.selectionRange;if(t!==n&&n&&n.ownerDocument&&sm(n.ownerDocument.documentElement,n)){if(r!==null&&Ru(n)){if(t=r.start,e=r.end,e===void 0&&(e=t),"selectionStart"in n)n.selectionStart=t,n.selectionEnd=Math.min(e,n.value.length);else if(e=(t=n.ownerDocument||document)&&t.defaultView||window,e.getSelection){e=e.getSelection();var i=n.textContent.length,a=Math.min(r.start,i);r=r.end===void 0?a:Math.min(r.end,i),!e.extend&&a>r&&(i=r,r=a,a=i),i=id(n,a);var o=id(n,r);i&&o&&(e.rangeCount!==1||e.anchorNode!==i.node||e.anchorOffset!==i.offset||e.focusNode!==o.node||e.focusOffset!==o.offset)&&(t=t.createRange(),t.setStart(i.node,i.offset),e.removeAllRanges(),a>r?(e.addRange(t),e.extend(o.node,o.offset)):(t.setEnd(o.node,o.offset),e.addRange(t)))}}for(t=[],e=n;e=e.parentNode;)e.nodeType===1&&t.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<t.length;n++)e=t[n],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var Zx=Jt&&"documentMode"in document&&11>=document.documentMode,cr=null,_l=null,yi=null,wl=!1;function ad(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;wl||cr==null||cr!==Ya(r)||(r=cr,"selectionStart"in r&&Ru(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),yi&&ji(yi,r)||(yi=r,r=ro(_l,"onSelect"),0<r.length&&(t=new Au("onSelect","select",null,t,n),e.push({event:t,listeners:r}),t.target=cr)))}function ha(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n["Webkit"+e]="webkit"+t,n["Moz"+e]="moz"+t,n}var dr={animationend:ha("Animation","AnimationEnd"),animationiteration:ha("Animation","AnimationIteration"),animationstart:ha("Animation","AnimationStart"),transitionend:ha("Transition","TransitionEnd")},ps={},um={};Jt&&(um=document.createElement("div").style,"AnimationEvent"in window||(delete dr.animationend.animation,delete dr.animationiteration.animation,delete dr.animationstart.animation),"TransitionEvent"in window||delete dr.transitionend.transition);function Lo(e){if(ps[e])return ps[e];if(!dr[e])return e;var t=dr[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in um)return ps[e]=t[n];return e}var cm=Lo("animationend"),dm=Lo("animationiteration"),pm=Lo("animationstart"),fm=Lo("transitionend"),mm=new Map,od="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function zn(e,t){mm.set(e,t),Jn(t,[e])}for(var fs=0;fs<od.length;fs++){var ms=od[fs],Jx=ms.toLowerCase(),e0=ms[0].toUpperCase()+ms.slice(1);zn(Jx,"on"+e0)}zn(cm,"onAnimationEnd");zn(dm,"onAnimationIteration");zn(pm,"onAnimationStart");zn("dblclick","onDoubleClick");zn("focusin","onFocus");zn("focusout","onBlur");zn(fm,"onTransitionEnd");Nr("onMouseEnter",["mouseout","mouseover"]);Nr("onMouseLeave",["mouseout","mouseover"]);Nr("onPointerEnter",["pointerout","pointerover"]);Nr("onPointerLeave",["pointerout","pointerover"]);Jn("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));Jn("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));Jn("onBeforeInput",["compositionend","keypress","textInput","paste"]);Jn("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));Jn("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));Jn("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var ci="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),t0=new Set("cancel close invalid load scroll toggle".split(" ").concat(ci));function sd(e,t,n){var r=e.type||"unknown-event";e.currentTarget=n,Jy(r,t,void 0,e),e.currentTarget=null}function hm(e,t){t=(t&4)!==0;for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;e:{var a=void 0;if(t)for(var o=r.length-1;0<=o;o--){var s=r[o],l=s.instance,u=s.currentTarget;if(s=s.listener,l!==a&&i.isPropagationStopped())break e;sd(i,s,u),a=l}else for(o=0;o<r.length;o++){if(s=r[o],l=s.instance,u=s.currentTarget,s=s.listener,l!==a&&i.isPropagationStopped())break e;sd(i,s,u),a=l}}}if(Za)throw e=gl,Za=!1,gl=null,e}function fe(e,t){var n=t[El];n===void 0&&(n=t[El]=new Set);var r=e+"__bubble";n.has(r)||(gm(t,e,2,!1),n.add(r))}function hs(e,t,n){var r=0;t&&(r|=4),gm(n,e,r,t)}var ga="_reactListening"+Math.random().toString(36).slice(2);function Ri(e){if(!e[ga]){e[ga]=!0,kf.forEach(function(n){n!=="selectionchange"&&(t0.has(n)||hs(n,!1,e),hs(n,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[ga]||(t[ga]=!0,hs("selectionchange",!1,t))}}function gm(e,t,n,r){switch(Jf(t)){case 1:var i=hx;break;case 4:i=gx;break;default:i=Pu}n=i.bind(null,t,n,e),i=void 0,!hl||t!=="touchstart"&&t!=="touchmove"&&t!=="wheel"||(i=!0),r?i!==void 0?e.addEventListener(t,n,{capture:!0,passive:i}):e.addEventListener(t,n,!0):i!==void 0?e.addEventListener(t,n,{passive:i}):e.addEventListener(t,n,!1)}function gs(e,t,n,r,i){var a=r;if(!(t&1)&&!(t&2)&&r!==null)e:for(;;){if(r===null)return;var o=r.tag;if(o===3||o===4){var s=r.stateNode.containerInfo;if(s===i||s.nodeType===8&&s.parentNode===i)break;if(o===4)for(o=r.return;o!==null;){var l=o.tag;if((l===3||l===4)&&(l=o.stateNode.containerInfo,l===i||l.nodeType===8&&l.parentNode===i))return;o=o.return}for(;s!==null;){if(o=qn(s),o===null)return;if(l=o.tag,l===5||l===6){r=a=o;continue e}s=s.parentNode}}r=r.return}Ff(function(){var u=a,d=Cu(n),c=[];e:{var p=mm.get(e);if(p!==void 0){var f=Au,y=e;switch(e){case"keypress":if(Da(n)===0)break e;case"keydown":case"keyup":f=Ax;break;case"focusin":y="focus",f=us;break;case"focusout":y="blur",f=us;break;case"beforeblur":case"afterblur":f=us;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":f=Xc;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":f=vx;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":f=Rx;break;case cm:case dm:case pm:f=bx;break;case fm:f=Ox;break;case"scroll":f=yx;break;case"wheel":f=Mx;break;case"copy":case"cut":case"paste":f=Sx;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":f=Gc}var v=(t&4)!==0,k=!v&&e==="scroll",m=v?p!==null?p+"Capture":null:p;v=[];for(var g=u,x;g!==null;){x=g;var C=x.stateNode;if(x.tag===5&&C!==null&&(x=C,m!==null&&(C=Ti(g,m),C!=null&&v.push(Ii(g,C,x)))),k)break;g=g.return}0<v.length&&(p=new f(p,y,null,n,d),c.push({event:p,listeners:v}))}}if(!(t&7)){e:{if(p=e==="mouseover"||e==="pointerover",f=e==="mouseout"||e==="pointerout",p&&n!==fl&&(y=n.relatedTarget||n.fromElement)&&(qn(y)||y[en]))break e;if((f||p)&&(p=d.window===d?d:(p=d.ownerDocument)?p.defaultView||p.parentWindow:window,f?(y=n.relatedTarget||n.toElement,f=u,y=y?qn(y):null,y!==null&&(k=er(y),y!==k||y.tag!==5&&y.tag!==6)&&(y=null)):(f=null,y=u),f!==y)){if(v=Xc,C="onMouseLeave",m="onMouseEnter",g="mouse",(e==="pointerout"||e==="pointerover")&&(v=Gc,C="onPointerLeave",m="onPointerEnter",g="pointer"),k=f==null?p:pr(f),x=y==null?p:pr(y),p=new v(C,g+"leave",f,n,d),p.target=k,p.relatedTarget=x,C=null,qn(d)===u&&(v=new v(m,g+"enter",y,n,d),v.target=x,v.relatedTarget=k,C=v),k=C,f&&y)t:{for(v=f,m=y,g=0,x=v;x;x=rr(x))g++;for(x=0,C=m;C;C=rr(C))x++;for(;0<g-x;)v=rr(v),g--;for(;0<x-g;)m=rr(m),x--;for(;g--;){if(v===m||m!==null&&v===m.alternate)break t;v=rr(v),m=rr(m)}v=null}else v=null;f!==null&&ld(c,p,f,v,!1),y!==null&&k!==null&&ld(c,k,y,v,!0)}}e:{if(p=u?pr(u):window,f=p.nodeName&&p.nodeName.toLowerCase(),f==="select"||f==="input"&&p.type==="file")var z=Hx;else if(ed(p))if(am)z=Xx;else{z=Qx;var S=$x}else(f=p.nodeName)&&f.toLowerCase()==="input"&&(p.type==="checkbox"||p.type==="radio")&&(z=Kx);if(z&&(z=z(e,u))){im(c,z,n,d);break e}S&&S(e,p,u),e==="focusout"&&(S=p._wrapperState)&&S.controlled&&p.type==="number"&&ll(p,"number",p.value)}switch(S=u?pr(u):window,e){case"focusin":(ed(S)||S.contentEditable==="true")&&(cr=S,_l=u,yi=null);break;case"focusout":yi=_l=cr=null;break;case"mousedown":wl=!0;break;case"contextmenu":case"mouseup":case"dragend":wl=!1,ad(c,n,d);break;case"selectionchange":if(Zx)break;case"keydown":case"keyup":ad(c,n,d)}var T;if(ju)e:{switch(e){case"compositionstart":var N="onCompositionStart";break e;case"compositionend":N="onCompositionEnd";break e;case"compositionupdate":N="onCompositionUpdate";break e}N=void 0}else ur?nm(e,n)&&(N="onCompositionEnd"):e==="keydown"&&n.keyCode===229&&(N="onCompositionStart");N&&(tm&&n.locale!=="ko"&&(ur||N!=="onCompositionStart"?N==="onCompositionEnd"&&ur&&(T=em()):(pn=d,Nu="value"in pn?pn.value:pn.textContent,ur=!0)),S=ro(u,N),0<S.length&&(N=new Yc(N,e,null,n,d),c.push({event:N,listeners:S}),T?N.data=T:(T=rm(n),T!==null&&(N.data=T)))),(T=qx?Wx(e,n):Bx(e,n))&&(u=ro(u,"onBeforeInput"),0<u.length&&(d=new Yc("onBeforeInput","beforeinput",null,n,d),c.push({event:d,listeners:u}),d.data=T))}hm(c,t)})}function Ii(e,t,n){return{instance:e,listener:t,currentTarget:n}}function ro(e,t){for(var n=t+"Capture",r=[];e!==null;){var i=e,a=i.stateNode;i.tag===5&&a!==null&&(i=a,a=Ti(e,n),a!=null&&r.unshift(Ii(e,a,i)),a=Ti(e,t),a!=null&&r.push(Ii(e,a,i))),e=e.return}return r}function rr(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function ld(e,t,n,r,i){for(var a=t._reactName,o=[];n!==null&&n!==r;){var s=n,l=s.alternate,u=s.stateNode;if(l!==null&&l===r)break;s.tag===5&&u!==null&&(s=u,i?(l=Ti(n,a),l!=null&&o.unshift(Ii(n,l,s))):i||(l=Ti(n,a),l!=null&&o.push(Ii(n,l,s)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var n0=/\r\n?/g,r0=/\u0000|\uFFFD/g;function ud(e){return(typeof e=="string"?e:""+e).replace(n0,`
`).replace(r0,"")}function ya(e,t,n){if(t=ud(t),ud(e)!==t&&n)throw Error(j(425))}function io(){}var bl=null,kl=null;function Sl(e,t){return e==="textarea"||e==="noscript"||typeof t.children=="string"||typeof t.children=="number"||typeof t.dangerouslySetInnerHTML=="object"&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var Cl=typeof setTimeout=="function"?setTimeout:void 0,i0=typeof clearTimeout=="function"?clearTimeout:void 0,cd=typeof Promise=="function"?Promise:void 0,a0=typeof queueMicrotask=="function"?queueMicrotask:typeof cd<"u"?function(e){return cd.resolve(null).then(e).catch(o0)}:Cl;function o0(e){setTimeout(function(){throw e})}function ys(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8)if(n=i.data,n==="/$"){if(r===0){e.removeChild(i),Ai(t);return}r--}else n!=="$"&&n!=="$?"&&n!=="$!"||r++;n=i}while(n);Ai(t)}function xn(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t==="$"||t==="$!"||t==="$?")break;if(t==="/$")return null}}return e}function dd(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="$"||n==="$!"||n==="$?"){if(t===0)return e;t--}else n==="/$"&&t++}e=e.previousSibling}return null}var Br=Math.random().toString(36).slice(2),qt="__reactFiber$"+Br,Oi="__reactProps$"+Br,en="__reactContainer$"+Br,El="__reactEvents$"+Br,s0="__reactListeners$"+Br,l0="__reactHandles$"+Br;function qn(e){var t=e[qt];if(t)return t;for(var n=e.parentNode;n;){if(t=n[en]||n[qt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=dd(e);e!==null;){if(n=e[qt])return n;e=dd(e)}return t}e=n,n=e.parentNode}return null}function Zi(e){return e=e[qt]||e[en],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function pr(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(j(33))}function jo(e){return e[Oi]||null}var zl=[],fr=-1;function Tn(e){return{current:e}}function me(e){0>fr||(e.current=zl[fr],zl[fr]=null,fr--)}function pe(e,t){fr++,zl[fr]=e.current,e.current=t}var En={},qe=Tn(En),Ge=Tn(!1),$n=En;function Ar(e,t){var n=e.type.contextTypes;if(!n)return En;var r=e.stateNode;if(r&&r.__reactInternalMemoizedUnmaskedChildContext===t)return r.__reactInternalMemoizedMaskedChildContext;var i={},a;for(a in n)i[a]=t[a];return r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=t,e.__reactInternalMemoizedMaskedChildContext=i),i}function Ze(e){return e=e.childContextTypes,e!=null}function ao(){me(Ge),me(qe)}function pd(e,t,n){if(qe.current!==En)throw Error(j(168));pe(qe,t),pe(Ge,n)}function ym(e,t,n){var r=e.stateNode;if(t=t.childContextTypes,typeof r.getChildContext!="function")return n;r=r.getChildContext();for(var i in r)if(!(i in t))throw Error(j(108,$y(e)||"Unknown",i));return xe({},n,r)}function oo(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||En,$n=qe.current,pe(qe,e),pe(Ge,Ge.current),!0}function fd(e,t,n){var r=e.stateNode;if(!r)throw Error(j(169));n?(e=ym(e,t,$n),r.__reactInternalMemoizedMergedChildContext=e,me(Ge),me(qe),pe(qe,e)):me(Ge),pe(Ge,n)}var Xt=null,Ro=!1,xs=!1;function xm(e){Xt===null?Xt=[e]:Xt.push(e)}function u0(e){Ro=!0,xm(e)}function Pn(){if(!xs&&Xt!==null){xs=!0;var e=0,t=se;try{var n=Xt;for(se=1;e<n.length;e++){var r=n[e];do r=r(!0);while(r!==null)}Xt=null,Ro=!1}catch(i){throw Xt!==null&&(Xt=Xt.slice(e+1)),Uf(Eu,Pn),i}finally{se=t,xs=!1}}return null}var mr=[],hr=0,so=null,lo=0,mt=[],ht=0,Qn=null,Yt=1,Gt="";function Dn(e,t){mr[hr++]=lo,mr[hr++]=so,so=e,lo=t}function vm(e,t,n){mt[ht++]=Yt,mt[ht++]=Gt,mt[ht++]=Qn,Qn=e;var r=Yt;e=Gt;var i=32-Lt(r)-1;r&=~(1<<i),n+=1;var a=32-Lt(t)+i;if(30<a){var o=i-i%5;a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,Yt=1<<32-Lt(t)+i|n<<i|r,Gt=a+e}else Yt=1<<a|n<<i|r,Gt=e}function Iu(e){e.return!==null&&(Dn(e,1),vm(e,1,0))}function Ou(e){for(;e===so;)so=mr[--hr],mr[hr]=null,lo=mr[--hr],mr[hr]=null;for(;e===Qn;)Qn=mt[--ht],mt[ht]=null,Gt=mt[--ht],mt[ht]=null,Yt=mt[--ht],mt[ht]=null}var ot=null,at=null,he=!1,Pt=null;function _m(e,t){var n=yt(5,null,null,0);n.elementType="DELETED",n.stateNode=t,n.return=e,t=e.deletions,t===null?(e.deletions=[n],e.flags|=16):t.push(n)}function md(e,t){switch(e.tag){case 5:var n=e.type;return t=t.nodeType!==1||n.toLowerCase()!==t.nodeName.toLowerCase()?null:t,t!==null?(e.stateNode=t,ot=e,at=xn(t.firstChild),!0):!1;case 6:return t=e.pendingProps===""||t.nodeType!==3?null:t,t!==null?(e.stateNode=t,ot=e,at=null,!0):!1;case 13:return t=t.nodeType!==8?null:t,t!==null?(n=Qn!==null?{id:Yt,overflow:Gt}:null,e.memoizedState={dehydrated:t,treeContext:n,retryLane:1073741824},n=yt(18,null,null,0),n.stateNode=t,n.return=e,e.child=n,ot=e,at=null,!0):!1;default:return!1}}function Tl(e){return(e.mode&1)!==0&&(e.flags&128)===0}function Pl(e){if(he){var t=at;if(t){var n=t;if(!md(e,t)){if(Tl(e))throw Error(j(418));t=xn(n.nextSibling);var r=ot;t&&md(e,t)?_m(r,n):(e.flags=e.flags&-4097|2,he=!1,ot=e)}}else{if(Tl(e))throw Error(j(418));e.flags=e.flags&-4097|2,he=!1,ot=e}}}function hd(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;ot=e}function xa(e){if(e!==ot)return!1;if(!he)return hd(e),he=!0,!1;var t;if((t=e.tag!==3)&&!(t=e.tag!==5)&&(t=e.type,t=t!=="head"&&t!=="body"&&!Sl(e.type,e.memoizedProps)),t&&(t=at)){if(Tl(e))throw wm(),Error(j(418));for(;t;)_m(e,t),t=xn(t.nextSibling)}if(hd(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(j(317));e:{for(e=e.nextSibling,t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="/$"){if(t===0){at=xn(e.nextSibling);break e}t--}else n!=="$"&&n!=="$!"&&n!=="$?"||t++}e=e.nextSibling}at=null}}else at=ot?xn(e.stateNode.nextSibling):null;return!0}function wm(){for(var e=at;e;)e=xn(e.nextSibling)}function Lr(){at=ot=null,he=!1}function Du(e){Pt===null?Pt=[e]:Pt.push(e)}var c0=rn.ReactCurrentBatchConfig;function Jr(e,t,n){if(e=n.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(j(309));var r=n.stateNode}if(!r)throw Error(j(147,e));var i=r,a=""+e;return t!==null&&t.ref!==null&&typeof t.ref=="function"&&t.ref._stringRef===a?t.ref:(t=function(o){var s=i.refs;o===null?delete s[a]:s[a]=o},t._stringRef=a,t)}if(typeof e!="string")throw Error(j(284));if(!n._owner)throw Error(j(290,e))}return e}function va(e,t){throw e=Object.prototype.toString.call(t),Error(j(31,e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e))}function gd(e){var t=e._init;return t(e._payload)}function bm(e){function t(m,g){if(e){var x=m.deletions;x===null?(m.deletions=[g],m.flags|=16):x.push(g)}}function n(m,g){if(!e)return null;for(;g!==null;)t(m,g),g=g.sibling;return null}function r(m,g){for(m=new Map;g!==null;)g.key!==null?m.set(g.key,g):m.set(g.index,g),g=g.sibling;return m}function i(m,g){return m=bn(m,g),m.index=0,m.sibling=null,m}function a(m,g,x){return m.index=x,e?(x=m.alternate,x!==null?(x=x.index,x<g?(m.flags|=2,g):x):(m.flags|=2,g)):(m.flags|=1048576,g)}function o(m){return e&&m.alternate===null&&(m.flags|=2),m}function s(m,g,x,C){return g===null||g.tag!==6?(g=Cs(x,m.mode,C),g.return=m,g):(g=i(g,x),g.return=m,g)}function l(m,g,x,C){var z=x.type;return z===lr?d(m,g,x.props.children,C,x.key):g!==null&&(g.elementType===z||typeof z=="object"&&z!==null&&z.$$typeof===ln&&gd(z)===g.type)?(C=i(g,x.props),C.ref=Jr(m,g,x),C.return=m,C):(C=Va(x.type,x.key,x.props,null,m.mode,C),C.ref=Jr(m,g,x),C.return=m,C)}function u(m,g,x,C){return g===null||g.tag!==4||g.stateNode.containerInfo!==x.containerInfo||g.stateNode.implementation!==x.implementation?(g=Es(x,m.mode,C),g.return=m,g):(g=i(g,x.children||[]),g.return=m,g)}function d(m,g,x,C,z){return g===null||g.tag!==7?(g=Hn(x,m.mode,C,z),g.return=m,g):(g=i(g,x),g.return=m,g)}function c(m,g,x){if(typeof g=="string"&&g!==""||typeof g=="number")return g=Cs(""+g,m.mode,x),g.return=m,g;if(typeof g=="object"&&g!==null){switch(g.$$typeof){case la:return x=Va(g.type,g.key,g.props,null,m.mode,x),x.ref=Jr(m,null,g),x.return=m,x;case sr:return g=Es(g,m.mode,x),g.return=m,g;case ln:var C=g._init;return c(m,C(g._payload),x)}if(li(g)||Kr(g))return g=Hn(g,m.mode,x,null),g.return=m,g;va(m,g)}return null}function p(m,g,x,C){var z=g!==null?g.key:null;if(typeof x=="string"&&x!==""||typeof x=="number")return z!==null?null:s(m,g,""+x,C);if(typeof x=="object"&&x!==null){switch(x.$$typeof){case la:return x.key===z?l(m,g,x,C):null;case sr:return x.key===z?u(m,g,x,C):null;case ln:return z=x._init,p(m,g,z(x._payload),C)}if(li(x)||Kr(x))return z!==null?null:d(m,g,x,C,null);va(m,x)}return null}function f(m,g,x,C,z){if(typeof C=="string"&&C!==""||typeof C=="number")return m=m.get(x)||null,s(g,m,""+C,z);if(typeof C=="object"&&C!==null){switch(C.$$typeof){case la:return m=m.get(C.key===null?x:C.key)||null,l(g,m,C,z);case sr:return m=m.get(C.key===null?x:C.key)||null,u(g,m,C,z);case ln:var S=C._init;return f(m,g,x,S(C._payload),z)}if(li(C)||Kr(C))return m=m.get(x)||null,d(g,m,C,z,null);va(g,C)}return null}function y(m,g,x,C){for(var z=null,S=null,T=g,N=g=0,F=null;T!==null&&N<x.length;N++){T.index>N?(F=T,T=null):F=T.sibling;var b=p(m,T,x[N],C);if(b===null){T===null&&(T=F);break}e&&T&&b.alternate===null&&t(m,T),g=a(b,g,N),S===null?z=b:S.sibling=b,S=b,T=F}if(N===x.length)return n(m,T),he&&Dn(m,N),z;if(T===null){for(;N<x.length;N++)T=c(m,x[N],C),T!==null&&(g=a(T,g,N),S===null?z=T:S.sibling=T,S=T);return he&&Dn(m,N),z}for(T=r(m,T);N<x.length;N++)F=f(T,m,N,x[N],C),F!==null&&(e&&F.alternate!==null&&T.delete(F.key===null?N:F.key),g=a(F,g,N),S===null?z=F:S.sibling=F,S=F);return e&&T.forEach(function(I){return t(m,I)}),he&&Dn(m,N),z}function v(m,g,x,C){var z=Kr(x);if(typeof z!="function")throw Error(j(150));if(x=z.call(x),x==null)throw Error(j(151));for(var S=z=null,T=g,N=g=0,F=null,b=x.next();T!==null&&!b.done;N++,b=x.next()){T.index>N?(F=T,T=null):F=T.sibling;var I=p(m,T,b.value,C);if(I===null){T===null&&(T=F);break}e&&T&&I.alternate===null&&t(m,T),g=a(I,g,N),S===null?z=I:S.sibling=I,S=I,T=F}if(b.done)return n(m,T),he&&Dn(m,N),z;if(T===null){for(;!b.done;N++,b=x.next())b=c(m,b.value,C),b!==null&&(g=a(b,g,N),S===null?z=b:S.sibling=b,S=b);return he&&Dn(m,N),z}for(T=r(m,T);!b.done;N++,b=x.next())b=f(T,m,N,b.value,C),b!==null&&(e&&b.alternate!==null&&T.delete(b.key===null?N:b.key),g=a(b,g,N),S===null?z=b:S.sibling=b,S=b);return e&&T.forEach(function(W){return t(m,W)}),he&&Dn(m,N),z}function k(m,g,x,C){if(typeof x=="object"&&x!==null&&x.type===lr&&x.key===null&&(x=x.props.children),typeof x=="object"&&x!==null){switch(x.$$typeof){case la:e:{for(var z=x.key,S=g;S!==null;){if(S.key===z){if(z=x.type,z===lr){if(S.tag===7){n(m,S.sibling),g=i(S,x.props.children),g.return=m,m=g;break e}}else if(S.elementType===z||typeof z=="object"&&z!==null&&z.$$typeof===ln&&gd(z)===S.type){n(m,S.sibling),g=i(S,x.props),g.ref=Jr(m,S,x),g.return=m,m=g;break e}n(m,S);break}else t(m,S);S=S.sibling}x.type===lr?(g=Hn(x.props.children,m.mode,C,x.key),g.return=m,m=g):(C=Va(x.type,x.key,x.props,null,m.mode,C),C.ref=Jr(m,g,x),C.return=m,m=C)}return o(m);case sr:e:{for(S=x.key;g!==null;){if(g.key===S)if(g.tag===4&&g.stateNode.containerInfo===x.containerInfo&&g.stateNode.implementation===x.implementation){n(m,g.sibling),g=i(g,x.children||[]),g.return=m,m=g;break e}else{n(m,g);break}else t(m,g);g=g.sibling}g=Es(x,m.mode,C),g.return=m,m=g}return o(m);case ln:return S=x._init,k(m,g,S(x._payload),C)}if(li(x))return y(m,g,x,C);if(Kr(x))return v(m,g,x,C);va(m,x)}return typeof x=="string"&&x!==""||typeof x=="number"?(x=""+x,g!==null&&g.tag===6?(n(m,g.sibling),g=i(g,x),g.return=m,m=g):(n(m,g),g=Cs(x,m.mode,C),g.return=m,m=g),o(m)):n(m,g)}return k}var jr=bm(!0),km=bm(!1),uo=Tn(null),co=null,gr=null,Mu=null;function Fu(){Mu=gr=co=null}function qu(e){var t=uo.current;me(uo),e._currentValue=t}function Nl(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)!==t?(e.childLanes|=t,r!==null&&(r.childLanes|=t)):r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t),e===n)break;e=e.return}}function Sr(e,t){co=e,Mu=gr=null,e=e.dependencies,e!==null&&e.firstContext!==null&&(e.lanes&t&&(Ye=!0),e.firstContext=null)}function vt(e){var t=e._currentValue;if(Mu!==e)if(e={context:e,memoizedValue:t,next:null},gr===null){if(co===null)throw Error(j(308));gr=e,co.dependencies={lanes:0,firstContext:e}}else gr=gr.next=e;return t}var Wn=null;function Wu(e){Wn===null?Wn=[e]:Wn.push(e)}function Sm(e,t,n,r){var i=t.interleaved;return i===null?(n.next=n,Wu(t)):(n.next=i.next,i.next=n),t.interleaved=n,tn(e,r)}function tn(e,t){e.lanes|=t;var n=e.alternate;for(n!==null&&(n.lanes|=t),n=e,e=e.return;e!==null;)e.childLanes|=t,n=e.alternate,n!==null&&(n.childLanes|=t),n=e,e=e.return;return n.tag===3?n.stateNode:null}var un=!1;function Bu(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function Cm(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function Zt(e,t){return{eventTime:e,lane:t,tag:0,payload:null,callback:null,next:null}}function vn(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,ae&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,tn(e,n)}return i=r.interleaved,i===null?(t.next=t,Wu(r)):(t.next=i.next,i.next=t),r.interleaved=t,tn(e,n)}function Ma(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,(n&4194240)!==0)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,zu(e,n)}}function yd(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,a=null;if(n=n.firstBaseUpdate,n!==null){do{var o={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};a===null?i=a=o:a=a.next=o,n=n.next}while(n!==null);a===null?i=a=t:a=a.next=t}else i=a=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:a,shared:r.shared,effects:r.effects},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}function po(e,t,n,r){var i=e.updateQueue;un=!1;var a=i.firstBaseUpdate,o=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var l=s,u=l.next;l.next=null,o===null?a=u:o.next=u,o=l;var d=e.alternate;d!==null&&(d=d.updateQueue,s=d.lastBaseUpdate,s!==o&&(s===null?d.firstBaseUpdate=u:s.next=u,d.lastBaseUpdate=l))}if(a!==null){var c=i.baseState;o=0,d=u=l=null,s=a;do{var p=s.lane,f=s.eventTime;if((r&p)===p){d!==null&&(d=d.next={eventTime:f,lane:0,tag:s.tag,payload:s.payload,callback:s.callback,next:null});e:{var y=e,v=s;switch(p=t,f=n,v.tag){case 1:if(y=v.payload,typeof y=="function"){c=y.call(f,c,p);break e}c=y;break e;case 3:y.flags=y.flags&-65537|128;case 0:if(y=v.payload,p=typeof y=="function"?y.call(f,c,p):y,p==null)break e;c=xe({},c,p);break e;case 2:un=!0}}s.callback!==null&&s.lane!==0&&(e.flags|=64,p=i.effects,p===null?i.effects=[s]:p.push(s))}else f={eventTime:f,lane:p,tag:s.tag,payload:s.payload,callback:s.callback,next:null},d===null?(u=d=f,l=c):d=d.next=f,o|=p;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;p=s,s=p.next,p.next=null,i.lastBaseUpdate=p,i.shared.pending=null}}while(!0);if(d===null&&(l=c),i.baseState=l,i.firstBaseUpdate=u,i.lastBaseUpdate=d,t=i.shared.interleaved,t!==null){i=t;do o|=i.lane,i=i.next;while(i!==t)}else a===null&&(i.shared.lanes=0);Xn|=o,e.lanes=o,e.memoizedState=c}}function xd(e,t,n){if(e=t.effects,t.effects=null,e!==null)for(t=0;t<e.length;t++){var r=e[t],i=r.callback;if(i!==null){if(r.callback=null,r=n,typeof i!="function")throw Error(j(191,i));i.call(r)}}}var Ji={},Bt=Tn(Ji),Di=Tn(Ji),Mi=Tn(Ji);function Bn(e){if(e===Ji)throw Error(j(174));return e}function Uu(e,t){switch(pe(Mi,t),pe(Di,e),pe(Bt,Ji),e=t.nodeType,e){case 9:case 11:t=(t=t.documentElement)?t.namespaceURI:cl(null,"");break;default:e=e===8?t.parentNode:t,t=e.namespaceURI||null,e=e.tagName,t=cl(t,e)}me(Bt),pe(Bt,t)}function Rr(){me(Bt),me(Di),me(Mi)}function Em(e){Bn(Mi.current);var t=Bn(Bt.current),n=cl(t,e.type);t!==n&&(pe(Di,e),pe(Bt,n))}function Vu(e){Di.current===e&&(me(Bt),me(Di))}var ge=Tn(0);function fo(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==void 0){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var vs=[];function Hu(){for(var e=0;e<vs.length;e++)vs[e]._workInProgressVersionPrimary=null;vs.length=0}var Fa=rn.ReactCurrentDispatcher,_s=rn.ReactCurrentBatchConfig,Kn=0,ye=null,ze=null,Pe=null,mo=!1,xi=!1,Fi=0,d0=0;function Ie(){throw Error(j(321))}function $u(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!It(e[n],t[n]))return!1;return!0}function Qu(e,t,n,r,i,a){if(Kn=a,ye=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,Fa.current=e===null||e.memoizedState===null?h0:g0,e=n(r,i),xi){a=0;do{if(xi=!1,Fi=0,25<=a)throw Error(j(301));a+=1,Pe=ze=null,t.updateQueue=null,Fa.current=y0,e=n(r,i)}while(xi)}if(Fa.current=ho,t=ze!==null&&ze.next!==null,Kn=0,Pe=ze=ye=null,mo=!1,t)throw Error(j(300));return e}function Ku(){var e=Fi!==0;return Fi=0,e}function Mt(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Pe===null?ye.memoizedState=Pe=e:Pe=Pe.next=e,Pe}function _t(){if(ze===null){var e=ye.alternate;e=e!==null?e.memoizedState:null}else e=ze.next;var t=Pe===null?ye.memoizedState:Pe.next;if(t!==null)Pe=t,ze=e;else{if(e===null)throw Error(j(310));ze=e,e={memoizedState:ze.memoizedState,baseState:ze.baseState,baseQueue:ze.baseQueue,queue:ze.queue,next:null},Pe===null?ye.memoizedState=Pe=e:Pe=Pe.next=e}return Pe}function qi(e,t){return typeof t=="function"?t(e):t}function ws(e){var t=_t(),n=t.queue;if(n===null)throw Error(j(311));n.lastRenderedReducer=e;var r=ze,i=r.baseQueue,a=n.pending;if(a!==null){if(i!==null){var o=i.next;i.next=a.next,a.next=o}r.baseQueue=i=a,n.pending=null}if(i!==null){a=i.next,r=r.baseState;var s=o=null,l=null,u=a;do{var d=u.lane;if((Kn&d)===d)l!==null&&(l=l.next={lane:0,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),r=u.hasEagerState?u.eagerState:e(r,u.action);else{var c={lane:d,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null};l===null?(s=l=c,o=r):l=l.next=c,ye.lanes|=d,Xn|=d}u=u.next}while(u!==null&&u!==a);l===null?o=r:l.next=s,It(r,t.memoizedState)||(Ye=!0),t.memoizedState=r,t.baseState=o,t.baseQueue=l,n.lastRenderedState=r}if(e=n.interleaved,e!==null){i=e;do a=i.lane,ye.lanes|=a,Xn|=a,i=i.next;while(i!==e)}else i===null&&(n.lanes=0);return[t.memoizedState,n.dispatch]}function bs(e){var t=_t(),n=t.queue;if(n===null)throw Error(j(311));n.lastRenderedReducer=e;var r=n.dispatch,i=n.pending,a=t.memoizedState;if(i!==null){n.pending=null;var o=i=i.next;do a=e(a,o.action),o=o.next;while(o!==i);It(a,t.memoizedState)||(Ye=!0),t.memoizedState=a,t.baseQueue===null&&(t.baseState=a),n.lastRenderedState=a}return[a,r]}function zm(){}function Tm(e,t){var n=ye,r=_t(),i=t(),a=!It(r.memoizedState,i);if(a&&(r.memoizedState=i,Ye=!0),r=r.queue,Xu(Am.bind(null,n,r,e),[e]),r.getSnapshot!==t||a||Pe!==null&&Pe.memoizedState.tag&1){if(n.flags|=2048,Wi(9,Nm.bind(null,n,r,i,t),void 0,null),Ae===null)throw Error(j(349));Kn&30||Pm(n,t,i)}return i}function Pm(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=ye.updateQueue,t===null?(t={lastEffect:null,stores:null},ye.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function Nm(e,t,n,r){t.value=n,t.getSnapshot=r,Lm(t)&&jm(e)}function Am(e,t,n){return n(function(){Lm(t)&&jm(e)})}function Lm(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!It(e,n)}catch{return!0}}function jm(e){var t=tn(e,1);t!==null&&jt(t,e,1,-1)}function vd(e){var t=Mt();return typeof e=="function"&&(e=e()),t.memoizedState=t.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:qi,lastRenderedState:e},t.queue=e,e=e.dispatch=m0.bind(null,ye,e),[t.memoizedState,e]}function Wi(e,t,n,r){return e={tag:e,create:t,destroy:n,deps:r,next:null},t=ye.updateQueue,t===null?(t={lastEffect:null,stores:null},ye.updateQueue=t,t.lastEffect=e.next=e):(n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e)),e}function Rm(){return _t().memoizedState}function qa(e,t,n,r){var i=Mt();ye.flags|=e,i.memoizedState=Wi(1|t,n,void 0,r===void 0?null:r)}function Io(e,t,n,r){var i=_t();r=r===void 0?null:r;var a=void 0;if(ze!==null){var o=ze.memoizedState;if(a=o.destroy,r!==null&&$u(r,o.deps)){i.memoizedState=Wi(t,n,a,r);return}}ye.flags|=e,i.memoizedState=Wi(1|t,n,a,r)}function _d(e,t){return qa(8390656,8,e,t)}function Xu(e,t){return Io(2048,8,e,t)}function Im(e,t){return Io(4,2,e,t)}function Om(e,t){return Io(4,4,e,t)}function Dm(e,t){if(typeof t=="function")return e=e(),t(e),function(){t(null)};if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function Mm(e,t,n){return n=n!=null?n.concat([e]):null,Io(4,4,Dm.bind(null,t,e),n)}function Yu(){}function Fm(e,t){var n=_t();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&$u(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function qm(e,t){var n=_t();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&$u(t,r[1])?r[0]:(e=e(),n.memoizedState=[e,t],e)}function Wm(e,t,n){return Kn&21?(It(n,t)||(n=$f(),ye.lanes|=n,Xn|=n,e.baseState=!0),t):(e.baseState&&(e.baseState=!1,Ye=!0),e.memoizedState=n)}function p0(e,t){var n=se;se=n!==0&&4>n?n:4,e(!0);var r=_s.transition;_s.transition={};try{e(!1),t()}finally{se=n,_s.transition=r}}function Bm(){return _t().memoizedState}function f0(e,t,n){var r=wn(e);if(n={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null},Um(e))Vm(t,n);else if(n=Sm(e,t,n,r),n!==null){var i=He();jt(n,e,r,i),Hm(n,t,r)}}function m0(e,t,n){var r=wn(e),i={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null};if(Um(e))Vm(t,i);else{var a=e.alternate;if(e.lanes===0&&(a===null||a.lanes===0)&&(a=t.lastRenderedReducer,a!==null))try{var o=t.lastRenderedState,s=a(o,n);if(i.hasEagerState=!0,i.eagerState=s,It(s,o)){var l=t.interleaved;l===null?(i.next=i,Wu(t)):(i.next=l.next,l.next=i),t.interleaved=i;return}}catch{}finally{}n=Sm(e,t,i,r),n!==null&&(i=He(),jt(n,e,r,i),Hm(n,t,r))}}function Um(e){var t=e.alternate;return e===ye||t!==null&&t===ye}function Vm(e,t){xi=mo=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function Hm(e,t,n){if(n&4194240){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,zu(e,n)}}var ho={readContext:vt,useCallback:Ie,useContext:Ie,useEffect:Ie,useImperativeHandle:Ie,useInsertionEffect:Ie,useLayoutEffect:Ie,useMemo:Ie,useReducer:Ie,useRef:Ie,useState:Ie,useDebugValue:Ie,useDeferredValue:Ie,useTransition:Ie,useMutableSource:Ie,useSyncExternalStore:Ie,useId:Ie,unstable_isNewReconciler:!1},h0={readContext:vt,useCallback:function(e,t){return Mt().memoizedState=[e,t===void 0?null:t],e},useContext:vt,useEffect:_d,useImperativeHandle:function(e,t,n){return n=n!=null?n.concat([e]):null,qa(4194308,4,Dm.bind(null,t,e),n)},useLayoutEffect:function(e,t){return qa(4194308,4,e,t)},useInsertionEffect:function(e,t){return qa(4,2,e,t)},useMemo:function(e,t){var n=Mt();return t=t===void 0?null:t,e=e(),n.memoizedState=[e,t],e},useReducer:function(e,t,n){var r=Mt();return t=n!==void 0?n(t):t,r.memoizedState=r.baseState=t,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:t},r.queue=e,e=e.dispatch=f0.bind(null,ye,e),[r.memoizedState,e]},useRef:function(e){var t=Mt();return e={current:e},t.memoizedState=e},useState:vd,useDebugValue:Yu,useDeferredValue:function(e){return Mt().memoizedState=e},useTransition:function(){var e=vd(!1),t=e[0];return e=p0.bind(null,e[1]),Mt().memoizedState=e,[t,e]},useMutableSource:function(){},useSyncExternalStore:function(e,t,n){var r=ye,i=Mt();if(he){if(n===void 0)throw Error(j(407));n=n()}else{if(n=t(),Ae===null)throw Error(j(349));Kn&30||Pm(r,t,n)}i.memoizedState=n;var a={value:n,getSnapshot:t};return i.queue=a,_d(Am.bind(null,r,a,e),[e]),r.flags|=2048,Wi(9,Nm.bind(null,r,a,n,t),void 0,null),n},useId:function(){var e=Mt(),t=Ae.identifierPrefix;if(he){var n=Gt,r=Yt;n=(r&~(1<<32-Lt(r)-1)).toString(32)+n,t=":"+t+"R"+n,n=Fi++,0<n&&(t+="H"+n.toString(32)),t+=":"}else n=d0++,t=":"+t+"r"+n.toString(32)+":";return e.memoizedState=t},unstable_isNewReconciler:!1},g0={readContext:vt,useCallback:Fm,useContext:vt,useEffect:Xu,useImperativeHandle:Mm,useInsertionEffect:Im,useLayoutEffect:Om,useMemo:qm,useReducer:ws,useRef:Rm,useState:function(){return ws(qi)},useDebugValue:Yu,useDeferredValue:function(e){var t=_t();return Wm(t,ze.memoizedState,e)},useTransition:function(){var e=ws(qi)[0],t=_t().memoizedState;return[e,t]},useMutableSource:zm,useSyncExternalStore:Tm,useId:Bm,unstable_isNewReconciler:!1},y0={readContext:vt,useCallback:Fm,useContext:vt,useEffect:Xu,useImperativeHandle:Mm,useInsertionEffect:Im,useLayoutEffect:Om,useMemo:qm,useReducer:bs,useRef:Rm,useState:function(){return bs(qi)},useDebugValue:Yu,useDeferredValue:function(e){var t=_t();return ze===null?t.memoizedState=e:Wm(t,ze.memoizedState,e)},useTransition:function(){var e=bs(qi)[0],t=_t().memoizedState;return[e,t]},useMutableSource:zm,useSyncExternalStore:Tm,useId:Bm,unstable_isNewReconciler:!1};function zt(e,t){if(e&&e.defaultProps){t=xe({},t),e=e.defaultProps;for(var n in e)t[n]===void 0&&(t[n]=e[n]);return t}return t}function Al(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:xe({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var Oo={isMounted:function(e){return(e=e._reactInternals)?er(e)===e:!1},enqueueSetState:function(e,t,n){e=e._reactInternals;var r=He(),i=wn(e),a=Zt(r,i);a.payload=t,n!=null&&(a.callback=n),t=vn(e,a,i),t!==null&&(jt(t,e,i,r),Ma(t,e,i))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=He(),i=wn(e),a=Zt(r,i);a.tag=1,a.payload=t,n!=null&&(a.callback=n),t=vn(e,a,i),t!==null&&(jt(t,e,i,r),Ma(t,e,i))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=He(),r=wn(e),i=Zt(n,r);i.tag=2,t!=null&&(i.callback=t),t=vn(e,i,r),t!==null&&(jt(t,e,r,n),Ma(t,e,r))}};function wd(e,t,n,r,i,a,o){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(r,a,o):t.prototype&&t.prototype.isPureReactComponent?!ji(n,r)||!ji(i,a):!0}function $m(e,t,n){var r=!1,i=En,a=t.contextType;return typeof a=="object"&&a!==null?a=vt(a):(i=Ze(t)?$n:qe.current,r=t.contextTypes,a=(r=r!=null)?Ar(e,i):En),t=new t(n,a),e.memoizedState=t.state!==null&&t.state!==void 0?t.state:null,t.updater=Oo,e.stateNode=t,t._reactInternals=e,r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=i,e.__reactInternalMemoizedMaskedChildContext=a),t}function bd(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps=="function"&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps=="function"&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&Oo.enqueueReplaceState(t,t.state,null)}function Ll(e,t,n,r){var i=e.stateNode;i.props=n,i.state=e.memoizedState,i.refs={},Bu(e);var a=t.contextType;typeof a=="object"&&a!==null?i.context=vt(a):(a=Ze(t)?$n:qe.current,i.context=Ar(e,a)),i.state=e.memoizedState,a=t.getDerivedStateFromProps,typeof a=="function"&&(Al(e,t,a,n),i.state=e.memoizedState),typeof t.getDerivedStateFromProps=="function"||typeof i.getSnapshotBeforeUpdate=="function"||typeof i.UNSAFE_componentWillMount!="function"&&typeof i.componentWillMount!="function"||(t=i.state,typeof i.componentWillMount=="function"&&i.componentWillMount(),typeof i.UNSAFE_componentWillMount=="function"&&i.UNSAFE_componentWillMount(),t!==i.state&&Oo.enqueueReplaceState(i,i.state,null),po(e,n,i,r),i.state=e.memoizedState),typeof i.componentDidMount=="function"&&(e.flags|=4194308)}function Ir(e,t){try{var n="",r=t;do n+=Hy(r),r=r.return;while(r);var i=n}catch(a){i=`
Error generating stack: `+a.message+`
`+a.stack}return{value:e,source:t,stack:i,digest:null}}function ks(e,t,n){return{value:e,source:null,stack:n??null,digest:t??null}}function jl(e,t){try{console.error(t.value)}catch(n){setTimeout(function(){throw n})}}var x0=typeof WeakMap=="function"?WeakMap:Map;function Qm(e,t,n){n=Zt(-1,n),n.tag=3,n.payload={element:null};var r=t.value;return n.callback=function(){yo||(yo=!0,Ul=r),jl(e,t)},n}function Km(e,t,n){n=Zt(-1,n),n.tag=3;var r=e.type.getDerivedStateFromError;if(typeof r=="function"){var i=t.value;n.payload=function(){return r(i)},n.callback=function(){jl(e,t)}}var a=e.stateNode;return a!==null&&typeof a.componentDidCatch=="function"&&(n.callback=function(){jl(e,t),typeof r!="function"&&(_n===null?_n=new Set([this]):_n.add(this));var o=t.stack;this.componentDidCatch(t.value,{componentStack:o!==null?o:""})}),n}function kd(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new x0;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(i.add(n),e=L0.bind(null,e,t,n),t.then(e,e))}function Sd(e){do{var t;if((t=e.tag===13)&&(t=e.memoizedState,t=t!==null?t.dehydrated!==null:!0),t)return e;e=e.return}while(e!==null);return null}function Cd(e,t,n,r,i){return e.mode&1?(e.flags|=65536,e.lanes=i,e):(e===t?e.flags|=65536:(e.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(t=Zt(-1,1),t.tag=2,vn(n,t,1))),n.lanes|=1),e)}var v0=rn.ReactCurrentOwner,Ye=!1;function Ue(e,t,n,r){t.child=e===null?km(t,null,n,r):jr(t,e.child,n,r)}function Ed(e,t,n,r,i){n=n.render;var a=t.ref;return Sr(t,i),r=Qu(e,t,n,r,a,i),n=Ku(),e!==null&&!Ye?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,nn(e,t,i)):(he&&n&&Iu(t),t.flags|=1,Ue(e,t,r,i),t.child)}function zd(e,t,n,r,i){if(e===null){var a=n.type;return typeof a=="function"&&!ic(a)&&a.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(t.tag=15,t.type=a,Xm(e,t,a,r,i)):(e=Va(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(a=e.child,!(e.lanes&i)){var o=a.memoizedProps;if(n=n.compare,n=n!==null?n:ji,n(o,r)&&e.ref===t.ref)return nn(e,t,i)}return t.flags|=1,e=bn(a,r),e.ref=t.ref,e.return=t,t.child=e}function Xm(e,t,n,r,i){if(e!==null){var a=e.memoizedProps;if(ji(a,r)&&e.ref===t.ref)if(Ye=!1,t.pendingProps=r=a,(e.lanes&i)!==0)e.flags&131072&&(Ye=!0);else return t.lanes=e.lanes,nn(e,t,i)}return Rl(e,t,n,r,i)}function Ym(e,t,n){var r=t.pendingProps,i=r.children,a=e!==null?e.memoizedState:null;if(r.mode==="hidden")if(!(t.mode&1))t.memoizedState={baseLanes:0,cachePool:null,transitions:null},pe(xr,rt),rt|=n;else{if(!(n&1073741824))return e=a!==null?a.baseLanes|n:n,t.lanes=t.childLanes=1073741824,t.memoizedState={baseLanes:e,cachePool:null,transitions:null},t.updateQueue=null,pe(xr,rt),rt|=e,null;t.memoizedState={baseLanes:0,cachePool:null,transitions:null},r=a!==null?a.baseLanes:n,pe(xr,rt),rt|=r}else a!==null?(r=a.baseLanes|n,t.memoizedState=null):r=n,pe(xr,rt),rt|=r;return Ue(e,t,i,n),t.child}function Gm(e,t){var n=t.ref;(e===null&&n!==null||e!==null&&e.ref!==n)&&(t.flags|=512,t.flags|=2097152)}function Rl(e,t,n,r,i){var a=Ze(n)?$n:qe.current;return a=Ar(t,a),Sr(t,i),n=Qu(e,t,n,r,a,i),r=Ku(),e!==null&&!Ye?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,nn(e,t,i)):(he&&r&&Iu(t),t.flags|=1,Ue(e,t,n,i),t.child)}function Td(e,t,n,r,i){if(Ze(n)){var a=!0;oo(t)}else a=!1;if(Sr(t,i),t.stateNode===null)Wa(e,t),$m(t,n,r),Ll(t,n,r,i),r=!0;else if(e===null){var o=t.stateNode,s=t.memoizedProps;o.props=s;var l=o.context,u=n.contextType;typeof u=="object"&&u!==null?u=vt(u):(u=Ze(n)?$n:qe.current,u=Ar(t,u));var d=n.getDerivedStateFromProps,c=typeof d=="function"||typeof o.getSnapshotBeforeUpdate=="function";c||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(s!==r||l!==u)&&bd(t,o,r,u),un=!1;var p=t.memoizedState;o.state=p,po(t,r,o,i),l=t.memoizedState,s!==r||p!==l||Ge.current||un?(typeof d=="function"&&(Al(t,n,d,r),l=t.memoizedState),(s=un||wd(t,n,s,r,p,l,u))?(c||typeof o.UNSAFE_componentWillMount!="function"&&typeof o.componentWillMount!="function"||(typeof o.componentWillMount=="function"&&o.componentWillMount(),typeof o.UNSAFE_componentWillMount=="function"&&o.UNSAFE_componentWillMount()),typeof o.componentDidMount=="function"&&(t.flags|=4194308)):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),o.props=r,o.state=l,o.context=u,r=s):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),r=!1)}else{o=t.stateNode,Cm(e,t),s=t.memoizedProps,u=t.type===t.elementType?s:zt(t.type,s),o.props=u,c=t.pendingProps,p=o.context,l=n.contextType,typeof l=="object"&&l!==null?l=vt(l):(l=Ze(n)?$n:qe.current,l=Ar(t,l));var f=n.getDerivedStateFromProps;(d=typeof f=="function"||typeof o.getSnapshotBeforeUpdate=="function")||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(s!==c||p!==l)&&bd(t,o,r,l),un=!1,p=t.memoizedState,o.state=p,po(t,r,o,i);var y=t.memoizedState;s!==c||p!==y||Ge.current||un?(typeof f=="function"&&(Al(t,n,f,r),y=t.memoizedState),(u=un||wd(t,n,u,r,p,y,l)||!1)?(d||typeof o.UNSAFE_componentWillUpdate!="function"&&typeof o.componentWillUpdate!="function"||(typeof o.componentWillUpdate=="function"&&o.componentWillUpdate(r,y,l),typeof o.UNSAFE_componentWillUpdate=="function"&&o.UNSAFE_componentWillUpdate(r,y,l)),typeof o.componentDidUpdate=="function"&&(t.flags|=4),typeof o.getSnapshotBeforeUpdate=="function"&&(t.flags|=1024)):(typeof o.componentDidUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=y),o.props=r,o.state=y,o.context=l,r=u):(typeof o.componentDidUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=1024),r=!1)}return Il(e,t,n,r,a,i)}function Il(e,t,n,r,i,a){Gm(e,t);var o=(t.flags&128)!==0;if(!r&&!o)return i&&fd(t,n,!1),nn(e,t,a);r=t.stateNode,v0.current=t;var s=o&&typeof n.getDerivedStateFromError!="function"?null:r.render();return t.flags|=1,e!==null&&o?(t.child=jr(t,e.child,null,a),t.child=jr(t,null,s,a)):Ue(e,t,s,a),t.memoizedState=r.state,i&&fd(t,n,!0),t.child}function Zm(e){var t=e.stateNode;t.pendingContext?pd(e,t.pendingContext,t.pendingContext!==t.context):t.context&&pd(e,t.context,!1),Uu(e,t.containerInfo)}function Pd(e,t,n,r,i){return Lr(),Du(i),t.flags|=256,Ue(e,t,n,r),t.child}var Ol={dehydrated:null,treeContext:null,retryLane:0};function Dl(e){return{baseLanes:e,cachePool:null,transitions:null}}function Jm(e,t,n){var r=t.pendingProps,i=ge.current,a=!1,o=(t.flags&128)!==0,s;if((s=o)||(s=e!==null&&e.memoizedState===null?!1:(i&2)!==0),s?(a=!0,t.flags&=-129):(e===null||e.memoizedState!==null)&&(i|=1),pe(ge,i&1),e===null)return Pl(t),e=t.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?(t.mode&1?e.data==="$!"?t.lanes=8:t.lanes=1073741824:t.lanes=1,null):(o=r.children,e=r.fallback,a?(r=t.mode,a=t.child,o={mode:"hidden",children:o},!(r&1)&&a!==null?(a.childLanes=0,a.pendingProps=o):a=Fo(o,r,0,null),e=Hn(e,r,n,null),a.return=t,e.return=t,a.sibling=e,t.child=a,t.child.memoizedState=Dl(n),t.memoizedState=Ol,e):Gu(t,o));if(i=e.memoizedState,i!==null&&(s=i.dehydrated,s!==null))return _0(e,t,o,r,s,i,n);if(a){a=r.fallback,o=t.mode,i=e.child,s=i.sibling;var l={mode:"hidden",children:r.children};return!(o&1)&&t.child!==i?(r=t.child,r.childLanes=0,r.pendingProps=l,t.deletions=null):(r=bn(i,l),r.subtreeFlags=i.subtreeFlags&14680064),s!==null?a=bn(s,a):(a=Hn(a,o,n,null),a.flags|=2),a.return=t,r.return=t,r.sibling=a,t.child=r,r=a,a=t.child,o=e.child.memoizedState,o=o===null?Dl(n):{baseLanes:o.baseLanes|n,cachePool:null,transitions:o.transitions},a.memoizedState=o,a.childLanes=e.childLanes&~n,t.memoizedState=Ol,r}return a=e.child,e=a.sibling,r=bn(a,{mode:"visible",children:r.children}),!(t.mode&1)&&(r.lanes=n),r.return=t,r.sibling=null,e!==null&&(n=t.deletions,n===null?(t.deletions=[e],t.flags|=16):n.push(e)),t.child=r,t.memoizedState=null,r}function Gu(e,t){return t=Fo({mode:"visible",children:t},e.mode,0,null),t.return=e,e.child=t}function _a(e,t,n,r){return r!==null&&Du(r),jr(t,e.child,null,n),e=Gu(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function _0(e,t,n,r,i,a,o){if(n)return t.flags&256?(t.flags&=-257,r=ks(Error(j(422))),_a(e,t,o,r)):t.memoizedState!==null?(t.child=e.child,t.flags|=128,null):(a=r.fallback,i=t.mode,r=Fo({mode:"visible",children:r.children},i,0,null),a=Hn(a,i,o,null),a.flags|=2,r.return=t,a.return=t,r.sibling=a,t.child=r,t.mode&1&&jr(t,e.child,null,o),t.child.memoizedState=Dl(o),t.memoizedState=Ol,a);if(!(t.mode&1))return _a(e,t,o,null);if(i.data==="$!"){if(r=i.nextSibling&&i.nextSibling.dataset,r)var s=r.dgst;return r=s,a=Error(j(419)),r=ks(a,r,void 0),_a(e,t,o,r)}if(s=(o&e.childLanes)!==0,Ye||s){if(r=Ae,r!==null){switch(o&-o){case 4:i=2;break;case 16:i=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:i=32;break;case 536870912:i=268435456;break;default:i=0}i=i&(r.suspendedLanes|o)?0:i,i!==0&&i!==a.retryLane&&(a.retryLane=i,tn(e,i),jt(r,e,i,-1))}return rc(),r=ks(Error(j(421))),_a(e,t,o,r)}return i.data==="$?"?(t.flags|=128,t.child=e.child,t=j0.bind(null,e),i._reactRetry=t,null):(e=a.treeContext,at=xn(i.nextSibling),ot=t,he=!0,Pt=null,e!==null&&(mt[ht++]=Yt,mt[ht++]=Gt,mt[ht++]=Qn,Yt=e.id,Gt=e.overflow,Qn=t),t=Gu(t,r.children),t.flags|=4096,t)}function Nd(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),Nl(e.return,t,n)}function Ss(e,t,n,r,i){var a=e.memoizedState;a===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i}:(a.isBackwards=t,a.rendering=null,a.renderingStartTime=0,a.last=r,a.tail=n,a.tailMode=i)}function eh(e,t,n){var r=t.pendingProps,i=r.revealOrder,a=r.tail;if(Ue(e,t,r.children,n),r=ge.current,r&2)r=r&1|2,t.flags|=128;else{if(e!==null&&e.flags&128)e:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&Nd(e,n,t);else if(e.tag===19)Nd(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}r&=1}if(pe(ge,r),!(t.mode&1))t.memoizedState=null;else switch(i){case"forwards":for(n=t.child,i=null;n!==null;)e=n.alternate,e!==null&&fo(e)===null&&(i=n),n=n.sibling;n=i,n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),Ss(t,!1,i,n,a);break;case"backwards":for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&fo(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}Ss(t,!0,n,null,a);break;case"together":Ss(t,!1,null,null,void 0);break;default:t.memoizedState=null}return t.child}function Wa(e,t){!(t.mode&1)&&e!==null&&(e.alternate=null,t.alternate=null,t.flags|=2)}function nn(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),Xn|=t.lanes,!(n&t.childLanes))return null;if(e!==null&&t.child!==e.child)throw Error(j(153));if(t.child!==null){for(e=t.child,n=bn(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=bn(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function w0(e,t,n){switch(t.tag){case 3:Zm(t),Lr();break;case 5:Em(t);break;case 1:Ze(t.type)&&oo(t);break;case 4:Uu(t,t.stateNode.containerInfo);break;case 10:var r=t.type._context,i=t.memoizedProps.value;pe(uo,r._currentValue),r._currentValue=i;break;case 13:if(r=t.memoizedState,r!==null)return r.dehydrated!==null?(pe(ge,ge.current&1),t.flags|=128,null):n&t.child.childLanes?Jm(e,t,n):(pe(ge,ge.current&1),e=nn(e,t,n),e!==null?e.sibling:null);pe(ge,ge.current&1);break;case 19:if(r=(n&t.childLanes)!==0,e.flags&128){if(r)return eh(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),pe(ge,ge.current),r)break;return null;case 22:case 23:return t.lanes=0,Ym(e,t,n)}return nn(e,t,n)}var th,Ml,nh,rh;th=function(e,t){for(var n=t.child;n!==null;){if(n.tag===5||n.tag===6)e.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===t)break;for(;n.sibling===null;){if(n.return===null||n.return===t)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};Ml=function(){};nh=function(e,t,n,r){var i=e.memoizedProps;if(i!==r){e=t.stateNode,Bn(Bt.current);var a=null;switch(n){case"input":i=ol(e,i),r=ol(e,r),a=[];break;case"select":i=xe({},i,{value:void 0}),r=xe({},r,{value:void 0}),a=[];break;case"textarea":i=ul(e,i),r=ul(e,r),a=[];break;default:typeof i.onClick!="function"&&typeof r.onClick=="function"&&(e.onclick=io)}dl(n,r);var o;n=null;for(u in i)if(!r.hasOwnProperty(u)&&i.hasOwnProperty(u)&&i[u]!=null)if(u==="style"){var s=i[u];for(o in s)s.hasOwnProperty(o)&&(n||(n={}),n[o]="")}else u!=="dangerouslySetInnerHTML"&&u!=="children"&&u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&u!=="autoFocus"&&(Ei.hasOwnProperty(u)?a||(a=[]):(a=a||[]).push(u,null));for(u in r){var l=r[u];if(s=i!=null?i[u]:void 0,r.hasOwnProperty(u)&&l!==s&&(l!=null||s!=null))if(u==="style")if(s){for(o in s)!s.hasOwnProperty(o)||l&&l.hasOwnProperty(o)||(n||(n={}),n[o]="");for(o in l)l.hasOwnProperty(o)&&s[o]!==l[o]&&(n||(n={}),n[o]=l[o])}else n||(a||(a=[]),a.push(u,n)),n=l;else u==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,s=s?s.__html:void 0,l!=null&&s!==l&&(a=a||[]).push(u,l)):u==="children"?typeof l!="string"&&typeof l!="number"||(a=a||[]).push(u,""+l):u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&(Ei.hasOwnProperty(u)?(l!=null&&u==="onScroll"&&fe("scroll",e),a||s===l||(a=[])):(a=a||[]).push(u,l))}n&&(a=a||[]).push("style",n);var u=a;(t.updateQueue=u)&&(t.flags|=4)}};rh=function(e,t,n,r){n!==r&&(t.flags|=4)};function ei(e,t){if(!he)switch(e.tailMode){case"hidden":t=e.tail;for(var n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null;break;case"collapsed":n=e.tail;for(var r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null}}function Oe(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&14680064,r|=i.flags&14680064,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function b0(e,t,n){var r=t.pendingProps;switch(Ou(t),t.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return Oe(t),null;case 1:return Ze(t.type)&&ao(),Oe(t),null;case 3:return r=t.stateNode,Rr(),me(Ge),me(qe),Hu(),r.pendingContext&&(r.context=r.pendingContext,r.pendingContext=null),(e===null||e.child===null)&&(xa(t)?t.flags|=4:e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,Pt!==null&&($l(Pt),Pt=null))),Ml(e,t),Oe(t),null;case 5:Vu(t);var i=Bn(Mi.current);if(n=t.type,e!==null&&t.stateNode!=null)nh(e,t,n,r,i),e.ref!==t.ref&&(t.flags|=512,t.flags|=2097152);else{if(!r){if(t.stateNode===null)throw Error(j(166));return Oe(t),null}if(e=Bn(Bt.current),xa(t)){r=t.stateNode,n=t.type;var a=t.memoizedProps;switch(r[qt]=t,r[Oi]=a,e=(t.mode&1)!==0,n){case"dialog":fe("cancel",r),fe("close",r);break;case"iframe":case"object":case"embed":fe("load",r);break;case"video":case"audio":for(i=0;i<ci.length;i++)fe(ci[i],r);break;case"source":fe("error",r);break;case"img":case"image":case"link":fe("error",r),fe("load",r);break;case"details":fe("toggle",r);break;case"input":Fc(r,a),fe("invalid",r);break;case"select":r._wrapperState={wasMultiple:!!a.multiple},fe("invalid",r);break;case"textarea":Wc(r,a),fe("invalid",r)}dl(n,a),i=null;for(var o in a)if(a.hasOwnProperty(o)){var s=a[o];o==="children"?typeof s=="string"?r.textContent!==s&&(a.suppressHydrationWarning!==!0&&ya(r.textContent,s,e),i=["children",s]):typeof s=="number"&&r.textContent!==""+s&&(a.suppressHydrationWarning!==!0&&ya(r.textContent,s,e),i=["children",""+s]):Ei.hasOwnProperty(o)&&s!=null&&o==="onScroll"&&fe("scroll",r)}switch(n){case"input":ua(r),qc(r,a,!0);break;case"textarea":ua(r),Bc(r);break;case"select":case"option":break;default:typeof a.onClick=="function"&&(r.onclick=io)}r=i,t.updateQueue=r,r!==null&&(t.flags|=4)}else{o=i.nodeType===9?i:i.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=Af(n)),e==="http://www.w3.org/1999/xhtml"?n==="script"?(e=o.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof r.is=="string"?e=o.createElement(n,{is:r.is}):(e=o.createElement(n),n==="select"&&(o=e,r.multiple?o.multiple=!0:r.size&&(o.size=r.size))):e=o.createElementNS(e,n),e[qt]=t,e[Oi]=r,th(e,t,!1,!1),t.stateNode=e;e:{switch(o=pl(n,r),n){case"dialog":fe("cancel",e),fe("close",e),i=r;break;case"iframe":case"object":case"embed":fe("load",e),i=r;break;case"video":case"audio":for(i=0;i<ci.length;i++)fe(ci[i],e);i=r;break;case"source":fe("error",e),i=r;break;case"img":case"image":case"link":fe("error",e),fe("load",e),i=r;break;case"details":fe("toggle",e),i=r;break;case"input":Fc(e,r),i=ol(e,r),fe("invalid",e);break;case"option":i=r;break;case"select":e._wrapperState={wasMultiple:!!r.multiple},i=xe({},r,{value:void 0}),fe("invalid",e);break;case"textarea":Wc(e,r),i=ul(e,r),fe("invalid",e);break;default:i=r}dl(n,i),s=i;for(a in s)if(s.hasOwnProperty(a)){var l=s[a];a==="style"?Rf(e,l):a==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,l!=null&&Lf(e,l)):a==="children"?typeof l=="string"?(n!=="textarea"||l!=="")&&zi(e,l):typeof l=="number"&&zi(e,""+l):a!=="suppressContentEditableWarning"&&a!=="suppressHydrationWarning"&&a!=="autoFocus"&&(Ei.hasOwnProperty(a)?l!=null&&a==="onScroll"&&fe("scroll",e):l!=null&&wu(e,a,l,o))}switch(n){case"input":ua(e),qc(e,r,!1);break;case"textarea":ua(e),Bc(e);break;case"option":r.value!=null&&e.setAttribute("value",""+Cn(r.value));break;case"select":e.multiple=!!r.multiple,a=r.value,a!=null?_r(e,!!r.multiple,a,!1):r.defaultValue!=null&&_r(e,!!r.multiple,r.defaultValue,!0);break;default:typeof i.onClick=="function"&&(e.onclick=io)}switch(n){case"button":case"input":case"select":case"textarea":r=!!r.autoFocus;break e;case"img":r=!0;break e;default:r=!1}}r&&(t.flags|=4)}t.ref!==null&&(t.flags|=512,t.flags|=2097152)}return Oe(t),null;case 6:if(e&&t.stateNode!=null)rh(e,t,e.memoizedProps,r);else{if(typeof r!="string"&&t.stateNode===null)throw Error(j(166));if(n=Bn(Mi.current),Bn(Bt.current),xa(t)){if(r=t.stateNode,n=t.memoizedProps,r[qt]=t,(a=r.nodeValue!==n)&&(e=ot,e!==null))switch(e.tag){case 3:ya(r.nodeValue,n,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&ya(r.nodeValue,n,(e.mode&1)!==0)}a&&(t.flags|=4)}else r=(n.nodeType===9?n:n.ownerDocument).createTextNode(r),r[qt]=t,t.stateNode=r}return Oe(t),null;case 13:if(me(ge),r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(he&&at!==null&&t.mode&1&&!(t.flags&128))wm(),Lr(),t.flags|=98560,a=!1;else if(a=xa(t),r!==null&&r.dehydrated!==null){if(e===null){if(!a)throw Error(j(318));if(a=t.memoizedState,a=a!==null?a.dehydrated:null,!a)throw Error(j(317));a[qt]=t}else Lr(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;Oe(t),a=!1}else Pt!==null&&($l(Pt),Pt=null),a=!0;if(!a)return t.flags&65536?t:null}return t.flags&128?(t.lanes=n,t):(r=r!==null,r!==(e!==null&&e.memoizedState!==null)&&r&&(t.child.flags|=8192,t.mode&1&&(e===null||ge.current&1?Te===0&&(Te=3):rc())),t.updateQueue!==null&&(t.flags|=4),Oe(t),null);case 4:return Rr(),Ml(e,t),e===null&&Ri(t.stateNode.containerInfo),Oe(t),null;case 10:return qu(t.type._context),Oe(t),null;case 17:return Ze(t.type)&&ao(),Oe(t),null;case 19:if(me(ge),a=t.memoizedState,a===null)return Oe(t),null;if(r=(t.flags&128)!==0,o=a.rendering,o===null)if(r)ei(a,!1);else{if(Te!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(o=fo(e),o!==null){for(t.flags|=128,ei(a,!1),r=o.updateQueue,r!==null&&(t.updateQueue=r,t.flags|=4),t.subtreeFlags=0,r=n,n=t.child;n!==null;)a=n,e=r,a.flags&=14680066,o=a.alternate,o===null?(a.childLanes=0,a.lanes=e,a.child=null,a.subtreeFlags=0,a.memoizedProps=null,a.memoizedState=null,a.updateQueue=null,a.dependencies=null,a.stateNode=null):(a.childLanes=o.childLanes,a.lanes=o.lanes,a.child=o.child,a.subtreeFlags=0,a.deletions=null,a.memoizedProps=o.memoizedProps,a.memoizedState=o.memoizedState,a.updateQueue=o.updateQueue,a.type=o.type,e=o.dependencies,a.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),n=n.sibling;return pe(ge,ge.current&1|2),t.child}e=e.sibling}a.tail!==null&&ke()>Or&&(t.flags|=128,r=!0,ei(a,!1),t.lanes=4194304)}else{if(!r)if(e=fo(o),e!==null){if(t.flags|=128,r=!0,n=e.updateQueue,n!==null&&(t.updateQueue=n,t.flags|=4),ei(a,!0),a.tail===null&&a.tailMode==="hidden"&&!o.alternate&&!he)return Oe(t),null}else 2*ke()-a.renderingStartTime>Or&&n!==1073741824&&(t.flags|=128,r=!0,ei(a,!1),t.lanes=4194304);a.isBackwards?(o.sibling=t.child,t.child=o):(n=a.last,n!==null?n.sibling=o:t.child=o,a.last=o)}return a.tail!==null?(t=a.tail,a.rendering=t,a.tail=t.sibling,a.renderingStartTime=ke(),t.sibling=null,n=ge.current,pe(ge,r?n&1|2:n&1),t):(Oe(t),null);case 22:case 23:return nc(),r=t.memoizedState!==null,e!==null&&e.memoizedState!==null!==r&&(t.flags|=8192),r&&t.mode&1?rt&1073741824&&(Oe(t),t.subtreeFlags&6&&(t.flags|=8192)):Oe(t),null;case 24:return null;case 25:return null}throw Error(j(156,t.tag))}function k0(e,t){switch(Ou(t),t.tag){case 1:return Ze(t.type)&&ao(),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return Rr(),me(Ge),me(qe),Hu(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 5:return Vu(t),null;case 13:if(me(ge),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(j(340));Lr()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return me(ge),null;case 4:return Rr(),null;case 10:return qu(t.type._context),null;case 22:case 23:return nc(),null;case 24:return null;default:return null}}var wa=!1,Me=!1,S0=typeof WeakSet=="function"?WeakSet:Set,U=null;function yr(e,t){var n=e.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(r){_e(e,t,r)}else n.current=null}function Fl(e,t,n){try{n()}catch(r){_e(e,t,r)}}var Ad=!1;function C0(e,t){if(bl=to,e=lm(),Ru(e)){if("selectionStart"in e)var n={start:e.selectionStart,end:e.selectionEnd};else e:{n=(n=e.ownerDocument)&&n.defaultView||window;var r=n.getSelection&&n.getSelection();if(r&&r.rangeCount!==0){n=r.anchorNode;var i=r.anchorOffset,a=r.focusNode;r=r.focusOffset;try{n.nodeType,a.nodeType}catch{n=null;break e}var o=0,s=-1,l=-1,u=0,d=0,c=e,p=null;t:for(;;){for(var f;c!==n||i!==0&&c.nodeType!==3||(s=o+i),c!==a||r!==0&&c.nodeType!==3||(l=o+r),c.nodeType===3&&(o+=c.nodeValue.length),(f=c.firstChild)!==null;)p=c,c=f;for(;;){if(c===e)break t;if(p===n&&++u===i&&(s=o),p===a&&++d===r&&(l=o),(f=c.nextSibling)!==null)break;c=p,p=c.parentNode}c=f}n=s===-1||l===-1?null:{start:s,end:l}}else n=null}n=n||{start:0,end:0}}else n=null;for(kl={focusedElem:e,selectionRange:n},to=!1,U=t;U!==null;)if(t=U,e=t.child,(t.subtreeFlags&1028)!==0&&e!==null)e.return=t,U=e;else for(;U!==null;){t=U;try{var y=t.alternate;if(t.flags&1024)switch(t.tag){case 0:case 11:case 15:break;case 1:if(y!==null){var v=y.memoizedProps,k=y.memoizedState,m=t.stateNode,g=m.getSnapshotBeforeUpdate(t.elementType===t.type?v:zt(t.type,v),k);m.__reactInternalSnapshotBeforeUpdate=g}break;case 3:var x=t.stateNode.containerInfo;x.nodeType===1?x.textContent="":x.nodeType===9&&x.documentElement&&x.removeChild(x.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(j(163))}}catch(C){_e(t,t.return,C)}if(e=t.sibling,e!==null){e.return=t.return,U=e;break}U=t.return}return y=Ad,Ad=!1,y}function vi(e,t,n){var r=t.updateQueue;if(r=r!==null?r.lastEffect:null,r!==null){var i=r=r.next;do{if((i.tag&e)===e){var a=i.destroy;i.destroy=void 0,a!==void 0&&Fl(t,n,a)}i=i.next}while(i!==r)}}function Do(e,t){if(t=t.updateQueue,t=t!==null?t.lastEffect:null,t!==null){var n=t=t.next;do{if((n.tag&e)===e){var r=n.create;n.destroy=r()}n=n.next}while(n!==t)}}function ql(e){var t=e.ref;if(t!==null){var n=e.stateNode;switch(e.tag){case 5:e=n;break;default:e=n}typeof t=="function"?t(e):t.current=e}}function ih(e){var t=e.alternate;t!==null&&(e.alternate=null,ih(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&(delete t[qt],delete t[Oi],delete t[El],delete t[s0],delete t[l0])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function ah(e){return e.tag===5||e.tag===3||e.tag===4}function Ld(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||ah(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function Wl(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.nodeType===8?n.parentNode.insertBefore(e,t):n.insertBefore(e,t):(n.nodeType===8?(t=n.parentNode,t.insertBefore(e,n)):(t=n,t.appendChild(e)),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=io));else if(r!==4&&(e=e.child,e!==null))for(Wl(e,t,n),e=e.sibling;e!==null;)Wl(e,t,n),e=e.sibling}function Bl(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.insertBefore(e,t):n.appendChild(e);else if(r!==4&&(e=e.child,e!==null))for(Bl(e,t,n),e=e.sibling;e!==null;)Bl(e,t,n),e=e.sibling}var Le=null,Tt=!1;function on(e,t,n){for(n=n.child;n!==null;)oh(e,t,n),n=n.sibling}function oh(e,t,n){if(Wt&&typeof Wt.onCommitFiberUnmount=="function")try{Wt.onCommitFiberUnmount(Po,n)}catch{}switch(n.tag){case 5:Me||yr(n,t);case 6:var r=Le,i=Tt;Le=null,on(e,t,n),Le=r,Tt=i,Le!==null&&(Tt?(e=Le,n=n.stateNode,e.nodeType===8?e.parentNode.removeChild(n):e.removeChild(n)):Le.removeChild(n.stateNode));break;case 18:Le!==null&&(Tt?(e=Le,n=n.stateNode,e.nodeType===8?ys(e.parentNode,n):e.nodeType===1&&ys(e,n),Ai(e)):ys(Le,n.stateNode));break;case 4:r=Le,i=Tt,Le=n.stateNode.containerInfo,Tt=!0,on(e,t,n),Le=r,Tt=i;break;case 0:case 11:case 14:case 15:if(!Me&&(r=n.updateQueue,r!==null&&(r=r.lastEffect,r!==null))){i=r=r.next;do{var a=i,o=a.destroy;a=a.tag,o!==void 0&&(a&2||a&4)&&Fl(n,t,o),i=i.next}while(i!==r)}on(e,t,n);break;case 1:if(!Me&&(yr(n,t),r=n.stateNode,typeof r.componentWillUnmount=="function"))try{r.props=n.memoizedProps,r.state=n.memoizedState,r.componentWillUnmount()}catch(s){_e(n,t,s)}on(e,t,n);break;case 21:on(e,t,n);break;case 22:n.mode&1?(Me=(r=Me)||n.memoizedState!==null,on(e,t,n),Me=r):on(e,t,n);break;default:on(e,t,n)}}function jd(e){var t=e.updateQueue;if(t!==null){e.updateQueue=null;var n=e.stateNode;n===null&&(n=e.stateNode=new S0),t.forEach(function(r){var i=R0.bind(null,e,r);n.has(r)||(n.add(r),r.then(i,i))})}}function bt(e,t){var n=t.deletions;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];try{var a=e,o=t,s=o;e:for(;s!==null;){switch(s.tag){case 5:Le=s.stateNode,Tt=!1;break e;case 3:Le=s.stateNode.containerInfo,Tt=!0;break e;case 4:Le=s.stateNode.containerInfo,Tt=!0;break e}s=s.return}if(Le===null)throw Error(j(160));oh(a,o,i),Le=null,Tt=!1;var l=i.alternate;l!==null&&(l.return=null),i.return=null}catch(u){_e(i,t,u)}}if(t.subtreeFlags&12854)for(t=t.child;t!==null;)sh(t,e),t=t.sibling}function sh(e,t){var n=e.alternate,r=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(bt(t,e),Dt(e),r&4){try{vi(3,e,e.return),Do(3,e)}catch(v){_e(e,e.return,v)}try{vi(5,e,e.return)}catch(v){_e(e,e.return,v)}}break;case 1:bt(t,e),Dt(e),r&512&&n!==null&&yr(n,n.return);break;case 5:if(bt(t,e),Dt(e),r&512&&n!==null&&yr(n,n.return),e.flags&32){var i=e.stateNode;try{zi(i,"")}catch(v){_e(e,e.return,v)}}if(r&4&&(i=e.stateNode,i!=null)){var a=e.memoizedProps,o=n!==null?n.memoizedProps:a,s=e.type,l=e.updateQueue;if(e.updateQueue=null,l!==null)try{s==="input"&&a.type==="radio"&&a.name!=null&&Pf(i,a),pl(s,o);var u=pl(s,a);for(o=0;o<l.length;o+=2){var d=l[o],c=l[o+1];d==="style"?Rf(i,c):d==="dangerouslySetInnerHTML"?Lf(i,c):d==="children"?zi(i,c):wu(i,d,c,u)}switch(s){case"input":sl(i,a);break;case"textarea":Nf(i,a);break;case"select":var p=i._wrapperState.wasMultiple;i._wrapperState.wasMultiple=!!a.multiple;var f=a.value;f!=null?_r(i,!!a.multiple,f,!1):p!==!!a.multiple&&(a.defaultValue!=null?_r(i,!!a.multiple,a.defaultValue,!0):_r(i,!!a.multiple,a.multiple?[]:"",!1))}i[Oi]=a}catch(v){_e(e,e.return,v)}}break;case 6:if(bt(t,e),Dt(e),r&4){if(e.stateNode===null)throw Error(j(162));i=e.stateNode,a=e.memoizedProps;try{i.nodeValue=a}catch(v){_e(e,e.return,v)}}break;case 3:if(bt(t,e),Dt(e),r&4&&n!==null&&n.memoizedState.isDehydrated)try{Ai(t.containerInfo)}catch(v){_e(e,e.return,v)}break;case 4:bt(t,e),Dt(e);break;case 13:bt(t,e),Dt(e),i=e.child,i.flags&8192&&(a=i.memoizedState!==null,i.stateNode.isHidden=a,!a||i.alternate!==null&&i.alternate.memoizedState!==null||(ec=ke())),r&4&&jd(e);break;case 22:if(d=n!==null&&n.memoizedState!==null,e.mode&1?(Me=(u=Me)||d,bt(t,e),Me=u):bt(t,e),Dt(e),r&8192){if(u=e.memoizedState!==null,(e.stateNode.isHidden=u)&&!d&&e.mode&1)for(U=e,d=e.child;d!==null;){for(c=U=d;U!==null;){switch(p=U,f=p.child,p.tag){case 0:case 11:case 14:case 15:vi(4,p,p.return);break;case 1:yr(p,p.return);var y=p.stateNode;if(typeof y.componentWillUnmount=="function"){r=p,n=p.return;try{t=r,y.props=t.memoizedProps,y.state=t.memoizedState,y.componentWillUnmount()}catch(v){_e(r,n,v)}}break;case 5:yr(p,p.return);break;case 22:if(p.memoizedState!==null){Id(c);continue}}f!==null?(f.return=p,U=f):Id(c)}d=d.sibling}e:for(d=null,c=e;;){if(c.tag===5){if(d===null){d=c;try{i=c.stateNode,u?(a=i.style,typeof a.setProperty=="function"?a.setProperty("display","none","important"):a.display="none"):(s=c.stateNode,l=c.memoizedProps.style,o=l!=null&&l.hasOwnProperty("display")?l.display:null,s.style.display=jf("display",o))}catch(v){_e(e,e.return,v)}}}else if(c.tag===6){if(d===null)try{c.stateNode.nodeValue=u?"":c.memoizedProps}catch(v){_e(e,e.return,v)}}else if((c.tag!==22&&c.tag!==23||c.memoizedState===null||c===e)&&c.child!==null){c.child.return=c,c=c.child;continue}if(c===e)break e;for(;c.sibling===null;){if(c.return===null||c.return===e)break e;d===c&&(d=null),c=c.return}d===c&&(d=null),c.sibling.return=c.return,c=c.sibling}}break;case 19:bt(t,e),Dt(e),r&4&&jd(e);break;case 21:break;default:bt(t,e),Dt(e)}}function Dt(e){var t=e.flags;if(t&2){try{e:{for(var n=e.return;n!==null;){if(ah(n)){var r=n;break e}n=n.return}throw Error(j(160))}switch(r.tag){case 5:var i=r.stateNode;r.flags&32&&(zi(i,""),r.flags&=-33);var a=Ld(e);Bl(e,a,i);break;case 3:case 4:var o=r.stateNode.containerInfo,s=Ld(e);Wl(e,s,o);break;default:throw Error(j(161))}}catch(l){_e(e,e.return,l)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function E0(e,t,n){U=e,lh(e)}function lh(e,t,n){for(var r=(e.mode&1)!==0;U!==null;){var i=U,a=i.child;if(i.tag===22&&r){var o=i.memoizedState!==null||wa;if(!o){var s=i.alternate,l=s!==null&&s.memoizedState!==null||Me;s=wa;var u=Me;if(wa=o,(Me=l)&&!u)for(U=i;U!==null;)o=U,l=o.child,o.tag===22&&o.memoizedState!==null?Od(i):l!==null?(l.return=o,U=l):Od(i);for(;a!==null;)U=a,lh(a),a=a.sibling;U=i,wa=s,Me=u}Rd(e)}else i.subtreeFlags&8772&&a!==null?(a.return=i,U=a):Rd(e)}}function Rd(e){for(;U!==null;){var t=U;if(t.flags&8772){var n=t.alternate;try{if(t.flags&8772)switch(t.tag){case 0:case 11:case 15:Me||Do(5,t);break;case 1:var r=t.stateNode;if(t.flags&4&&!Me)if(n===null)r.componentDidMount();else{var i=t.elementType===t.type?n.memoizedProps:zt(t.type,n.memoizedProps);r.componentDidUpdate(i,n.memoizedState,r.__reactInternalSnapshotBeforeUpdate)}var a=t.updateQueue;a!==null&&xd(t,a,r);break;case 3:var o=t.updateQueue;if(o!==null){if(n=null,t.child!==null)switch(t.child.tag){case 5:n=t.child.stateNode;break;case 1:n=t.child.stateNode}xd(t,o,n)}break;case 5:var s=t.stateNode;if(n===null&&t.flags&4){n=s;var l=t.memoizedProps;switch(t.type){case"button":case"input":case"select":case"textarea":l.autoFocus&&n.focus();break;case"img":l.src&&(n.src=l.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(t.memoizedState===null){var u=t.alternate;if(u!==null){var d=u.memoizedState;if(d!==null){var c=d.dehydrated;c!==null&&Ai(c)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(j(163))}Me||t.flags&512&&ql(t)}catch(p){_e(t,t.return,p)}}if(t===e){U=null;break}if(n=t.sibling,n!==null){n.return=t.return,U=n;break}U=t.return}}function Id(e){for(;U!==null;){var t=U;if(t===e){U=null;break}var n=t.sibling;if(n!==null){n.return=t.return,U=n;break}U=t.return}}function Od(e){for(;U!==null;){var t=U;try{switch(t.tag){case 0:case 11:case 15:var n=t.return;try{Do(4,t)}catch(l){_e(t,n,l)}break;case 1:var r=t.stateNode;if(typeof r.componentDidMount=="function"){var i=t.return;try{r.componentDidMount()}catch(l){_e(t,i,l)}}var a=t.return;try{ql(t)}catch(l){_e(t,a,l)}break;case 5:var o=t.return;try{ql(t)}catch(l){_e(t,o,l)}}}catch(l){_e(t,t.return,l)}if(t===e){U=null;break}var s=t.sibling;if(s!==null){s.return=t.return,U=s;break}U=t.return}}var z0=Math.ceil,go=rn.ReactCurrentDispatcher,Zu=rn.ReactCurrentOwner,xt=rn.ReactCurrentBatchConfig,ae=0,Ae=null,Ce=null,je=0,rt=0,xr=Tn(0),Te=0,Bi=null,Xn=0,Mo=0,Ju=0,_i=null,Xe=null,ec=0,Or=1/0,Kt=null,yo=!1,Ul=null,_n=null,ba=!1,fn=null,xo=0,wi=0,Vl=null,Ba=-1,Ua=0;function He(){return ae&6?ke():Ba!==-1?Ba:Ba=ke()}function wn(e){return e.mode&1?ae&2&&je!==0?je&-je:c0.transition!==null?(Ua===0&&(Ua=$f()),Ua):(e=se,e!==0||(e=window.event,e=e===void 0?16:Jf(e.type)),e):1}function jt(e,t,n,r){if(50<wi)throw wi=0,Vl=null,Error(j(185));Yi(e,n,r),(!(ae&2)||e!==Ae)&&(e===Ae&&(!(ae&2)&&(Mo|=n),Te===4&&dn(e,je)),Je(e,r),n===1&&ae===0&&!(t.mode&1)&&(Or=ke()+500,Ro&&Pn()))}function Je(e,t){var n=e.callbackNode;cx(e,t);var r=eo(e,e===Ae?je:0);if(r===0)n!==null&&Hc(n),e.callbackNode=null,e.callbackPriority=0;else if(t=r&-r,e.callbackPriority!==t){if(n!=null&&Hc(n),t===1)e.tag===0?u0(Dd.bind(null,e)):xm(Dd.bind(null,e)),a0(function(){!(ae&6)&&Pn()}),n=null;else{switch(Qf(r)){case 1:n=Eu;break;case 4:n=Vf;break;case 16:n=Ja;break;case 536870912:n=Hf;break;default:n=Ja}n=gh(n,uh.bind(null,e))}e.callbackPriority=t,e.callbackNode=n}}function uh(e,t){if(Ba=-1,Ua=0,ae&6)throw Error(j(327));var n=e.callbackNode;if(Cr()&&e.callbackNode!==n)return null;var r=eo(e,e===Ae?je:0);if(r===0)return null;if(r&30||r&e.expiredLanes||t)t=vo(e,r);else{t=r;var i=ae;ae|=2;var a=dh();(Ae!==e||je!==t)&&(Kt=null,Or=ke()+500,Vn(e,t));do try{N0();break}catch(s){ch(e,s)}while(!0);Fu(),go.current=a,ae=i,Ce!==null?t=0:(Ae=null,je=0,t=Te)}if(t!==0){if(t===2&&(i=yl(e),i!==0&&(r=i,t=Hl(e,i))),t===1)throw n=Bi,Vn(e,0),dn(e,r),Je(e,ke()),n;if(t===6)dn(e,r);else{if(i=e.current.alternate,!(r&30)&&!T0(i)&&(t=vo(e,r),t===2&&(a=yl(e),a!==0&&(r=a,t=Hl(e,a))),t===1))throw n=Bi,Vn(e,0),dn(e,r),Je(e,ke()),n;switch(e.finishedWork=i,e.finishedLanes=r,t){case 0:case 1:throw Error(j(345));case 2:Mn(e,Xe,Kt);break;case 3:if(dn(e,r),(r&130023424)===r&&(t=ec+500-ke(),10<t)){if(eo(e,0)!==0)break;if(i=e.suspendedLanes,(i&r)!==r){He(),e.pingedLanes|=e.suspendedLanes&i;break}e.timeoutHandle=Cl(Mn.bind(null,e,Xe,Kt),t);break}Mn(e,Xe,Kt);break;case 4:if(dn(e,r),(r&4194240)===r)break;for(t=e.eventTimes,i=-1;0<r;){var o=31-Lt(r);a=1<<o,o=t[o],o>i&&(i=o),r&=~a}if(r=i,r=ke()-r,r=(120>r?120:480>r?480:1080>r?1080:1920>r?1920:3e3>r?3e3:4320>r?4320:1960*z0(r/1960))-r,10<r){e.timeoutHandle=Cl(Mn.bind(null,e,Xe,Kt),r);break}Mn(e,Xe,Kt);break;case 5:Mn(e,Xe,Kt);break;default:throw Error(j(329))}}}return Je(e,ke()),e.callbackNode===n?uh.bind(null,e):null}function Hl(e,t){var n=_i;return e.current.memoizedState.isDehydrated&&(Vn(e,t).flags|=256),e=vo(e,t),e!==2&&(t=Xe,Xe=n,t!==null&&$l(t)),e}function $l(e){Xe===null?Xe=e:Xe.push.apply(Xe,e)}function T0(e){for(var t=e;;){if(t.flags&16384){var n=t.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var r=0;r<n.length;r++){var i=n[r],a=i.getSnapshot;i=i.value;try{if(!It(a(),i))return!1}catch{return!1}}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function dn(e,t){for(t&=~Ju,t&=~Mo,e.suspendedLanes|=t,e.pingedLanes&=~t,e=e.expirationTimes;0<t;){var n=31-Lt(t),r=1<<n;e[n]=-1,t&=~r}}function Dd(e){if(ae&6)throw Error(j(327));Cr();var t=eo(e,0);if(!(t&1))return Je(e,ke()),null;var n=vo(e,t);if(e.tag!==0&&n===2){var r=yl(e);r!==0&&(t=r,n=Hl(e,r))}if(n===1)throw n=Bi,Vn(e,0),dn(e,t),Je(e,ke()),n;if(n===6)throw Error(j(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,Mn(e,Xe,Kt),Je(e,ke()),null}function tc(e,t){var n=ae;ae|=1;try{return e(t)}finally{ae=n,ae===0&&(Or=ke()+500,Ro&&Pn())}}function Yn(e){fn!==null&&fn.tag===0&&!(ae&6)&&Cr();var t=ae;ae|=1;var n=xt.transition,r=se;try{if(xt.transition=null,se=1,e)return e()}finally{se=r,xt.transition=n,ae=t,!(ae&6)&&Pn()}}function nc(){rt=xr.current,me(xr)}function Vn(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(n!==-1&&(e.timeoutHandle=-1,i0(n)),Ce!==null)for(n=Ce.return;n!==null;){var r=n;switch(Ou(r),r.tag){case 1:r=r.type.childContextTypes,r!=null&&ao();break;case 3:Rr(),me(Ge),me(qe),Hu();break;case 5:Vu(r);break;case 4:Rr();break;case 13:me(ge);break;case 19:me(ge);break;case 10:qu(r.type._context);break;case 22:case 23:nc()}n=n.return}if(Ae=e,Ce=e=bn(e.current,null),je=rt=t,Te=0,Bi=null,Ju=Mo=Xn=0,Xe=_i=null,Wn!==null){for(t=0;t<Wn.length;t++)if(n=Wn[t],r=n.interleaved,r!==null){n.interleaved=null;var i=r.next,a=n.pending;if(a!==null){var o=a.next;a.next=i,r.next=o}n.pending=r}Wn=null}return e}function ch(e,t){do{var n=Ce;try{if(Fu(),Fa.current=ho,mo){for(var r=ye.memoizedState;r!==null;){var i=r.queue;i!==null&&(i.pending=null),r=r.next}mo=!1}if(Kn=0,Pe=ze=ye=null,xi=!1,Fi=0,Zu.current=null,n===null||n.return===null){Te=1,Bi=t,Ce=null;break}e:{var a=e,o=n.return,s=n,l=t;if(t=je,s.flags|=32768,l!==null&&typeof l=="object"&&typeof l.then=="function"){var u=l,d=s,c=d.tag;if(!(d.mode&1)&&(c===0||c===11||c===15)){var p=d.alternate;p?(d.updateQueue=p.updateQueue,d.memoizedState=p.memoizedState,d.lanes=p.lanes):(d.updateQueue=null,d.memoizedState=null)}var f=Sd(o);if(f!==null){f.flags&=-257,Cd(f,o,s,a,t),f.mode&1&&kd(a,u,t),t=f,l=u;var y=t.updateQueue;if(y===null){var v=new Set;v.add(l),t.updateQueue=v}else y.add(l);break e}else{if(!(t&1)){kd(a,u,t),rc();break e}l=Error(j(426))}}else if(he&&s.mode&1){var k=Sd(o);if(k!==null){!(k.flags&65536)&&(k.flags|=256),Cd(k,o,s,a,t),Du(Ir(l,s));break e}}a=l=Ir(l,s),Te!==4&&(Te=2),_i===null?_i=[a]:_i.push(a),a=o;do{switch(a.tag){case 3:a.flags|=65536,t&=-t,a.lanes|=t;var m=Qm(a,l,t);yd(a,m);break e;case 1:s=l;var g=a.type,x=a.stateNode;if(!(a.flags&128)&&(typeof g.getDerivedStateFromError=="function"||x!==null&&typeof x.componentDidCatch=="function"&&(_n===null||!_n.has(x)))){a.flags|=65536,t&=-t,a.lanes|=t;var C=Km(a,s,t);yd(a,C);break e}}a=a.return}while(a!==null)}fh(n)}catch(z){t=z,Ce===n&&n!==null&&(Ce=n=n.return);continue}break}while(!0)}function dh(){var e=go.current;return go.current=ho,e===null?ho:e}function rc(){(Te===0||Te===3||Te===2)&&(Te=4),Ae===null||!(Xn&268435455)&&!(Mo&268435455)||dn(Ae,je)}function vo(e,t){var n=ae;ae|=2;var r=dh();(Ae!==e||je!==t)&&(Kt=null,Vn(e,t));do try{P0();break}catch(i){ch(e,i)}while(!0);if(Fu(),ae=n,go.current=r,Ce!==null)throw Error(j(261));return Ae=null,je=0,Te}function P0(){for(;Ce!==null;)ph(Ce)}function N0(){for(;Ce!==null&&!tx();)ph(Ce)}function ph(e){var t=hh(e.alternate,e,rt);e.memoizedProps=e.pendingProps,t===null?fh(e):Ce=t,Zu.current=null}function fh(e){var t=e;do{var n=t.alternate;if(e=t.return,t.flags&32768){if(n=k0(n,t),n!==null){n.flags&=32767,Ce=n;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{Te=6,Ce=null;return}}else if(n=b0(n,t,rt),n!==null){Ce=n;return}if(t=t.sibling,t!==null){Ce=t;return}Ce=t=e}while(t!==null);Te===0&&(Te=5)}function Mn(e,t,n){var r=se,i=xt.transition;try{xt.transition=null,se=1,A0(e,t,n,r)}finally{xt.transition=i,se=r}return null}function A0(e,t,n,r){do Cr();while(fn!==null);if(ae&6)throw Error(j(327));n=e.finishedWork;var i=e.finishedLanes;if(n===null)return null;if(e.finishedWork=null,e.finishedLanes=0,n===e.current)throw Error(j(177));e.callbackNode=null,e.callbackPriority=0;var a=n.lanes|n.childLanes;if(dx(e,a),e===Ae&&(Ce=Ae=null,je=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||ba||(ba=!0,gh(Ja,function(){return Cr(),null})),a=(n.flags&15990)!==0,n.subtreeFlags&15990||a){a=xt.transition,xt.transition=null;var o=se;se=1;var s=ae;ae|=4,Zu.current=null,C0(e,n),sh(n,e),Gx(kl),to=!!bl,kl=bl=null,e.current=n,E0(n),nx(),ae=s,se=o,xt.transition=a}else e.current=n;if(ba&&(ba=!1,fn=e,xo=i),a=e.pendingLanes,a===0&&(_n=null),ax(n.stateNode),Je(e,ke()),t!==null)for(r=e.onRecoverableError,n=0;n<t.length;n++)i=t[n],r(i.value,{componentStack:i.stack,digest:i.digest});if(yo)throw yo=!1,e=Ul,Ul=null,e;return xo&1&&e.tag!==0&&Cr(),a=e.pendingLanes,a&1?e===Vl?wi++:(wi=0,Vl=e):wi=0,Pn(),null}function Cr(){if(fn!==null){var e=Qf(xo),t=xt.transition,n=se;try{if(xt.transition=null,se=16>e?16:e,fn===null)var r=!1;else{if(e=fn,fn=null,xo=0,ae&6)throw Error(j(331));var i=ae;for(ae|=4,U=e.current;U!==null;){var a=U,o=a.child;if(U.flags&16){var s=a.deletions;if(s!==null){for(var l=0;l<s.length;l++){var u=s[l];for(U=u;U!==null;){var d=U;switch(d.tag){case 0:case 11:case 15:vi(8,d,a)}var c=d.child;if(c!==null)c.return=d,U=c;else for(;U!==null;){d=U;var p=d.sibling,f=d.return;if(ih(d),d===u){U=null;break}if(p!==null){p.return=f,U=p;break}U=f}}}var y=a.alternate;if(y!==null){var v=y.child;if(v!==null){y.child=null;do{var k=v.sibling;v.sibling=null,v=k}while(v!==null)}}U=a}}if(a.subtreeFlags&2064&&o!==null)o.return=a,U=o;else e:for(;U!==null;){if(a=U,a.flags&2048)switch(a.tag){case 0:case 11:case 15:vi(9,a,a.return)}var m=a.sibling;if(m!==null){m.return=a.return,U=m;break e}U=a.return}}var g=e.current;for(U=g;U!==null;){o=U;var x=o.child;if(o.subtreeFlags&2064&&x!==null)x.return=o,U=x;else e:for(o=g;U!==null;){if(s=U,s.flags&2048)try{switch(s.tag){case 0:case 11:case 15:Do(9,s)}}catch(z){_e(s,s.return,z)}if(s===o){U=null;break e}var C=s.sibling;if(C!==null){C.return=s.return,U=C;break e}U=s.return}}if(ae=i,Pn(),Wt&&typeof Wt.onPostCommitFiberRoot=="function")try{Wt.onPostCommitFiberRoot(Po,e)}catch{}r=!0}return r}finally{se=n,xt.transition=t}}return!1}function Md(e,t,n){t=Ir(n,t),t=Qm(e,t,1),e=vn(e,t,1),t=He(),e!==null&&(Yi(e,1,t),Je(e,t))}function _e(e,t,n){if(e.tag===3)Md(e,e,n);else for(;t!==null;){if(t.tag===3){Md(t,e,n);break}else if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError=="function"||typeof r.componentDidCatch=="function"&&(_n===null||!_n.has(r))){e=Ir(n,e),e=Km(t,e,1),t=vn(t,e,1),e=He(),t!==null&&(Yi(t,1,e),Je(t,e));break}}t=t.return}}function L0(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),t=He(),e.pingedLanes|=e.suspendedLanes&n,Ae===e&&(je&n)===n&&(Te===4||Te===3&&(je&130023424)===je&&500>ke()-ec?Vn(e,0):Ju|=n),Je(e,t)}function mh(e,t){t===0&&(e.mode&1?(t=pa,pa<<=1,!(pa&130023424)&&(pa=4194304)):t=1);var n=He();e=tn(e,t),e!==null&&(Yi(e,t,n),Je(e,n))}function j0(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),mh(e,n)}function R0(e,t){var n=0;switch(e.tag){case 13:var r=e.stateNode,i=e.memoizedState;i!==null&&(n=i.retryLane);break;case 19:r=e.stateNode;break;default:throw Error(j(314))}r!==null&&r.delete(t),mh(e,n)}var hh;hh=function(e,t,n){if(e!==null)if(e.memoizedProps!==t.pendingProps||Ge.current)Ye=!0;else{if(!(e.lanes&n)&&!(t.flags&128))return Ye=!1,w0(e,t,n);Ye=!!(e.flags&131072)}else Ye=!1,he&&t.flags&1048576&&vm(t,lo,t.index);switch(t.lanes=0,t.tag){case 2:var r=t.type;Wa(e,t),e=t.pendingProps;var i=Ar(t,qe.current);Sr(t,n),i=Qu(null,t,r,e,i,n);var a=Ku();return t.flags|=1,typeof i=="object"&&i!==null&&typeof i.render=="function"&&i.$$typeof===void 0?(t.tag=1,t.memoizedState=null,t.updateQueue=null,Ze(r)?(a=!0,oo(t)):a=!1,t.memoizedState=i.state!==null&&i.state!==void 0?i.state:null,Bu(t),i.updater=Oo,t.stateNode=i,i._reactInternals=t,Ll(t,r,e,n),t=Il(null,t,r,!0,a,n)):(t.tag=0,he&&a&&Iu(t),Ue(null,t,i,n),t=t.child),t;case 16:r=t.elementType;e:{switch(Wa(e,t),e=t.pendingProps,i=r._init,r=i(r._payload),t.type=r,i=t.tag=O0(r),e=zt(r,e),i){case 0:t=Rl(null,t,r,e,n);break e;case 1:t=Td(null,t,r,e,n);break e;case 11:t=Ed(null,t,r,e,n);break e;case 14:t=zd(null,t,r,zt(r.type,e),n);break e}throw Error(j(306,r,""))}return t;case 0:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:zt(r,i),Rl(e,t,r,i,n);case 1:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:zt(r,i),Td(e,t,r,i,n);case 3:e:{if(Zm(t),e===null)throw Error(j(387));r=t.pendingProps,a=t.memoizedState,i=a.element,Cm(e,t),po(t,r,null,n);var o=t.memoizedState;if(r=o.element,a.isDehydrated)if(a={element:r,isDehydrated:!1,cache:o.cache,pendingSuspenseBoundaries:o.pendingSuspenseBoundaries,transitions:o.transitions},t.updateQueue.baseState=a,t.memoizedState=a,t.flags&256){i=Ir(Error(j(423)),t),t=Pd(e,t,r,n,i);break e}else if(r!==i){i=Ir(Error(j(424)),t),t=Pd(e,t,r,n,i);break e}else for(at=xn(t.stateNode.containerInfo.firstChild),ot=t,he=!0,Pt=null,n=km(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(Lr(),r===i){t=nn(e,t,n);break e}Ue(e,t,r,n)}t=t.child}return t;case 5:return Em(t),e===null&&Pl(t),r=t.type,i=t.pendingProps,a=e!==null?e.memoizedProps:null,o=i.children,Sl(r,i)?o=null:a!==null&&Sl(r,a)&&(t.flags|=32),Gm(e,t),Ue(e,t,o,n),t.child;case 6:return e===null&&Pl(t),null;case 13:return Jm(e,t,n);case 4:return Uu(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=jr(t,null,r,n):Ue(e,t,r,n),t.child;case 11:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:zt(r,i),Ed(e,t,r,i,n);case 7:return Ue(e,t,t.pendingProps,n),t.child;case 8:return Ue(e,t,t.pendingProps.children,n),t.child;case 12:return Ue(e,t,t.pendingProps.children,n),t.child;case 10:e:{if(r=t.type._context,i=t.pendingProps,a=t.memoizedProps,o=i.value,pe(uo,r._currentValue),r._currentValue=o,a!==null)if(It(a.value,o)){if(a.children===i.children&&!Ge.current){t=nn(e,t,n);break e}}else for(a=t.child,a!==null&&(a.return=t);a!==null;){var s=a.dependencies;if(s!==null){o=a.child;for(var l=s.firstContext;l!==null;){if(l.context===r){if(a.tag===1){l=Zt(-1,n&-n),l.tag=2;var u=a.updateQueue;if(u!==null){u=u.shared;var d=u.pending;d===null?l.next=l:(l.next=d.next,d.next=l),u.pending=l}}a.lanes|=n,l=a.alternate,l!==null&&(l.lanes|=n),Nl(a.return,n,t),s.lanes|=n;break}l=l.next}}else if(a.tag===10)o=a.type===t.type?null:a.child;else if(a.tag===18){if(o=a.return,o===null)throw Error(j(341));o.lanes|=n,s=o.alternate,s!==null&&(s.lanes|=n),Nl(o,n,t),o=a.sibling}else o=a.child;if(o!==null)o.return=a;else for(o=a;o!==null;){if(o===t){o=null;break}if(a=o.sibling,a!==null){a.return=o.return,o=a;break}o=o.return}a=o}Ue(e,t,i.children,n),t=t.child}return t;case 9:return i=t.type,r=t.pendingProps.children,Sr(t,n),i=vt(i),r=r(i),t.flags|=1,Ue(e,t,r,n),t.child;case 14:return r=t.type,i=zt(r,t.pendingProps),i=zt(r.type,i),zd(e,t,r,i,n);case 15:return Xm(e,t,t.type,t.pendingProps,n);case 17:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:zt(r,i),Wa(e,t),t.tag=1,Ze(r)?(e=!0,oo(t)):e=!1,Sr(t,n),$m(t,r,i),Ll(t,r,i,n),Il(null,t,r,!0,e,n);case 19:return eh(e,t,n);case 22:return Ym(e,t,n)}throw Error(j(156,t.tag))};function gh(e,t){return Uf(e,t)}function I0(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function yt(e,t,n,r){return new I0(e,t,n,r)}function ic(e){return e=e.prototype,!(!e||!e.isReactComponent)}function O0(e){if(typeof e=="function")return ic(e)?1:0;if(e!=null){if(e=e.$$typeof,e===ku)return 11;if(e===Su)return 14}return 2}function bn(e,t){var n=e.alternate;return n===null?(n=yt(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&14680064,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}function Va(e,t,n,r,i,a){var o=2;if(r=e,typeof e=="function")ic(e)&&(o=1);else if(typeof e=="string")o=5;else e:switch(e){case lr:return Hn(n.children,i,a,t);case bu:o=8,i|=8;break;case nl:return e=yt(12,n,t,i|2),e.elementType=nl,e.lanes=a,e;case rl:return e=yt(13,n,t,i),e.elementType=rl,e.lanes=a,e;case il:return e=yt(19,n,t,i),e.elementType=il,e.lanes=a,e;case Ef:return Fo(n,i,a,t);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case Sf:o=10;break e;case Cf:o=9;break e;case ku:o=11;break e;case Su:o=14;break e;case ln:o=16,r=null;break e}throw Error(j(130,e==null?e:typeof e,""))}return t=yt(o,n,t,i),t.elementType=e,t.type=r,t.lanes=a,t}function Hn(e,t,n,r){return e=yt(7,e,r,t),e.lanes=n,e}function Fo(e,t,n,r){return e=yt(22,e,r,t),e.elementType=Ef,e.lanes=n,e.stateNode={isHidden:!1},e}function Cs(e,t,n){return e=yt(6,e,null,t),e.lanes=n,e}function Es(e,t,n){return t=yt(4,e.children!==null?e.children:[],e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}function D0(e,t,n,r,i){this.tag=t,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=os(0),this.expirationTimes=os(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=os(0),this.identifierPrefix=r,this.onRecoverableError=i,this.mutableSourceEagerHydrationData=null}function ac(e,t,n,r,i,a,o,s,l){return e=new D0(e,t,n,s,l),t===1?(t=1,a===!0&&(t|=8)):t=0,a=yt(3,null,null,t),e.current=a,a.stateNode=e,a.memoizedState={element:r,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},Bu(a),e}function M0(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:sr,key:r==null?null:""+r,children:e,containerInfo:t,implementation:n}}function yh(e){if(!e)return En;e=e._reactInternals;e:{if(er(e)!==e||e.tag!==1)throw Error(j(170));var t=e;do{switch(t.tag){case 3:t=t.stateNode.context;break e;case 1:if(Ze(t.type)){t=t.stateNode.__reactInternalMemoizedMergedChildContext;break e}}t=t.return}while(t!==null);throw Error(j(171))}if(e.tag===1){var n=e.type;if(Ze(n))return ym(e,n,t)}return t}function xh(e,t,n,r,i,a,o,s,l){return e=ac(n,r,!0,e,i,a,o,s,l),e.context=yh(null),n=e.current,r=He(),i=wn(n),a=Zt(r,i),a.callback=t??null,vn(n,a,i),e.current.lanes=i,Yi(e,i,r),Je(e,r),e}function qo(e,t,n,r){var i=t.current,a=He(),o=wn(i);return n=yh(n),t.context===null?t.context=n:t.pendingContext=n,t=Zt(a,o),t.payload={element:e},r=r===void 0?null:r,r!==null&&(t.callback=r),e=vn(i,t,o),e!==null&&(jt(e,i,o,a),Ma(e,i,o)),o}function _o(e){if(e=e.current,!e.child)return null;switch(e.child.tag){case 5:return e.child.stateNode;default:return e.child.stateNode}}function Fd(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function oc(e,t){Fd(e,t),(e=e.alternate)&&Fd(e,t)}function F0(){return null}var vh=typeof reportError=="function"?reportError:function(e){console.error(e)};function sc(e){this._internalRoot=e}Wo.prototype.render=sc.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(j(409));qo(e,t,null,null)};Wo.prototype.unmount=sc.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;Yn(function(){qo(null,e,null,null)}),t[en]=null}};function Wo(e){this._internalRoot=e}Wo.prototype.unstable_scheduleHydration=function(e){if(e){var t=Yf();e={blockedOn:null,target:e,priority:t};for(var n=0;n<cn.length&&t!==0&&t<cn[n].priority;n++);cn.splice(n,0,e),n===0&&Zf(e)}};function lc(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function Bo(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function qd(){}function q0(e,t,n,r,i){if(i){if(typeof r=="function"){var a=r;r=function(){var u=_o(o);a.call(u)}}var o=xh(t,r,e,0,null,!1,!1,"",qd);return e._reactRootContainer=o,e[en]=o.current,Ri(e.nodeType===8?e.parentNode:e),Yn(),o}for(;i=e.lastChild;)e.removeChild(i);if(typeof r=="function"){var s=r;r=function(){var u=_o(l);s.call(u)}}var l=ac(e,0,!1,null,null,!1,!1,"",qd);return e._reactRootContainer=l,e[en]=l.current,Ri(e.nodeType===8?e.parentNode:e),Yn(function(){qo(t,l,n,r)}),l}function Uo(e,t,n,r,i){var a=n._reactRootContainer;if(a){var o=a;if(typeof i=="function"){var s=i;i=function(){var l=_o(o);s.call(l)}}qo(t,o,e,i)}else o=q0(n,t,e,i,r);return _o(o)}Kf=function(e){switch(e.tag){case 3:var t=e.stateNode;if(t.current.memoizedState.isDehydrated){var n=ui(t.pendingLanes);n!==0&&(zu(t,n|1),Je(t,ke()),!(ae&6)&&(Or=ke()+500,Pn()))}break;case 13:Yn(function(){var r=tn(e,1);if(r!==null){var i=He();jt(r,e,1,i)}}),oc(e,1)}};Tu=function(e){if(e.tag===13){var t=tn(e,134217728);if(t!==null){var n=He();jt(t,e,134217728,n)}oc(e,134217728)}};Xf=function(e){if(e.tag===13){var t=wn(e),n=tn(e,t);if(n!==null){var r=He();jt(n,e,t,r)}oc(e,t)}};Yf=function(){return se};Gf=function(e,t){var n=se;try{return se=e,t()}finally{se=n}};ml=function(e,t,n){switch(t){case"input":if(sl(e,n),t=n.name,n.type==="radio"&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+t)+'][type="radio"]'),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var i=jo(r);if(!i)throw Error(j(90));Tf(r),sl(r,i)}}}break;case"textarea":Nf(e,n);break;case"select":t=n.value,t!=null&&_r(e,!!n.multiple,t,!1)}};Df=tc;Mf=Yn;var W0={usingClientEntryPoint:!1,Events:[Zi,pr,jo,If,Of,tc]},ti={findFiberByHostInstance:qn,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},B0={bundleType:ti.bundleType,version:ti.version,rendererPackageName:ti.rendererPackageName,rendererConfig:ti.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:rn.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=Wf(e),e===null?null:e.stateNode},findFiberByHostInstance:ti.findFiberByHostInstance||F0,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var ka=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!ka.isDisabled&&ka.supportsFiber)try{Po=ka.inject(B0),Wt=ka}catch{}}ut.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=W0;ut.createPortal=function(e,t){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!lc(t))throw Error(j(200));return M0(e,t,null,n)};ut.createRoot=function(e,t){if(!lc(e))throw Error(j(299));var n=!1,r="",i=vh;return t!=null&&(t.unstable_strictMode===!0&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onRecoverableError!==void 0&&(i=t.onRecoverableError)),t=ac(e,1,!1,null,null,n,!1,r,i),e[en]=t.current,Ri(e.nodeType===8?e.parentNode:e),new sc(t)};ut.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var t=e._reactInternals;if(t===void 0)throw typeof e.render=="function"?Error(j(188)):(e=Object.keys(e).join(","),Error(j(268,e)));return e=Wf(t),e=e===null?null:e.stateNode,e};ut.flushSync=function(e){return Yn(e)};ut.hydrate=function(e,t,n){if(!Bo(t))throw Error(j(200));return Uo(null,e,t,!0,n)};ut.hydrateRoot=function(e,t,n){if(!lc(e))throw Error(j(405));var r=n!=null&&n.hydratedSources||null,i=!1,a="",o=vh;if(n!=null&&(n.unstable_strictMode===!0&&(i=!0),n.identifierPrefix!==void 0&&(a=n.identifierPrefix),n.onRecoverableError!==void 0&&(o=n.onRecoverableError)),t=xh(t,null,e,1,n??null,i,!1,a,o),e[en]=t.current,Ri(e),r)for(e=0;e<r.length;e++)n=r[e],i=n._getVersion,i=i(n._source),t.mutableSourceEagerHydrationData==null?t.mutableSourceEagerHydrationData=[n,i]:t.mutableSourceEagerHydrationData.push(n,i);return new Wo(t)};ut.render=function(e,t,n){if(!Bo(t))throw Error(j(200));return Uo(null,e,t,!1,n)};ut.unmountComponentAtNode=function(e){if(!Bo(e))throw Error(j(40));return e._reactRootContainer?(Yn(function(){Uo(null,null,e,!1,function(){e._reactRootContainer=null,e[en]=null})}),!0):!1};ut.unstable_batchedUpdates=tc;ut.unstable_renderSubtreeIntoContainer=function(e,t,n,r){if(!Bo(n))throw Error(j(200));if(e==null||e._reactInternals===void 0)throw Error(j(38));return Uo(e,t,n,!1,r)};ut.version="18.3.1-next-f1338f8080-20240426";function _h(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(_h)}catch(e){console.error(e)}}_h(),_f.exports=ut;var U0=_f.exports,Wd=U0;el.createRoot=Wd.createRoot,el.hydrateRoot=Wd.hydrateRoot;/**
 * @remix-run/router v1.23.2
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Ui(){return Ui=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Ui.apply(this,arguments)}var mn;(function(e){e.Pop="POP",e.Push="PUSH",e.Replace="REPLACE"})(mn||(mn={}));const Bd="popstate";function V0(e){e===void 0&&(e={});function t(r,i){let{pathname:a,search:o,hash:s}=r.location;return Ql("",{pathname:a,search:o,hash:s},i.state&&i.state.usr||null,i.state&&i.state.key||"default")}function n(r,i){return typeof i=="string"?i:wo(i)}return $0(t,n,null,e)}function we(e,t){if(e===!1||e===null||typeof e>"u")throw new Error(t)}function uc(e,t){if(!e){typeof console<"u"&&console.warn(t);try{throw new Error(t)}catch{}}}function H0(){return Math.random().toString(36).substr(2,8)}function Ud(e,t){return{usr:e.state,key:e.key,idx:t}}function Ql(e,t,n,r){return n===void 0&&(n=null),Ui({pathname:typeof e=="string"?e:e.pathname,search:"",hash:""},typeof t=="string"?Ur(t):t,{state:n,key:t&&t.key||r||H0()})}function wo(e){let{pathname:t="/",search:n="",hash:r=""}=e;return n&&n!=="?"&&(t+=n.charAt(0)==="?"?n:"?"+n),r&&r!=="#"&&(t+=r.charAt(0)==="#"?r:"#"+r),t}function Ur(e){let t={};if(e){let n=e.indexOf("#");n>=0&&(t.hash=e.substr(n),e=e.substr(0,n));let r=e.indexOf("?");r>=0&&(t.search=e.substr(r),e=e.substr(0,r)),e&&(t.pathname=e)}return t}function $0(e,t,n,r){r===void 0&&(r={});let{window:i=document.defaultView,v5Compat:a=!1}=r,o=i.history,s=mn.Pop,l=null,u=d();u==null&&(u=0,o.replaceState(Ui({},o.state,{idx:u}),""));function d(){return(o.state||{idx:null}).idx}function c(){s=mn.Pop;let k=d(),m=k==null?null:k-u;u=k,l&&l({action:s,location:v.location,delta:m})}function p(k,m){s=mn.Push;let g=Ql(v.location,k,m);u=d()+1;let x=Ud(g,u),C=v.createHref(g);try{o.pushState(x,"",C)}catch(z){if(z instanceof DOMException&&z.name==="DataCloneError")throw z;i.location.assign(C)}a&&l&&l({action:s,location:v.location,delta:1})}function f(k,m){s=mn.Replace;let g=Ql(v.location,k,m);u=d();let x=Ud(g,u),C=v.createHref(g);o.replaceState(x,"",C),a&&l&&l({action:s,location:v.location,delta:0})}function y(k){let m=i.location.origin!=="null"?i.location.origin:i.location.href,g=typeof k=="string"?k:wo(k);return g=g.replace(/ $/,"%20"),we(m,"No window.location.(origin|href) available to create URL for href: "+g),new URL(g,m)}let v={get action(){return s},get location(){return e(i,o)},listen(k){if(l)throw new Error("A history only accepts one active listener");return i.addEventListener(Bd,c),l=k,()=>{i.removeEventListener(Bd,c),l=null}},createHref(k){return t(i,k)},createURL:y,encodeLocation(k){let m=y(k);return{pathname:m.pathname,search:m.search,hash:m.hash}},push:p,replace:f,go(k){return o.go(k)}};return v}var Vd;(function(e){e.data="data",e.deferred="deferred",e.redirect="redirect",e.error="error"})(Vd||(Vd={}));function Q0(e,t,n){return n===void 0&&(n="/"),K0(e,t,n)}function K0(e,t,n,r){let i=typeof t=="string"?Ur(t):t,a=Dr(i.pathname||"/",n);if(a==null)return null;let o=wh(e);X0(o);let s=null;for(let l=0;s==null&&l<o.length;++l){let u=o1(a);s=i1(o[l],u)}return s}function wh(e,t,n,r){t===void 0&&(t=[]),n===void 0&&(n=[]),r===void 0&&(r="");let i=(a,o,s)=>{let l={relativePath:s===void 0?a.path||"":s,caseSensitive:a.caseSensitive===!0,childrenIndex:o,route:a};l.relativePath.startsWith("/")&&(we(l.relativePath.startsWith(r),'Absolute route path "'+l.relativePath+'" nested under path '+('"'+r+'" is not valid. An absolute child route path ')+"must start with the combined path of all its parent routes."),l.relativePath=l.relativePath.slice(r.length));let u=kn([r,l.relativePath]),d=n.concat(l);a.children&&a.children.length>0&&(we(a.index!==!0,"Index routes must not have child routes. Please remove "+('all child routes from route path "'+u+'".')),wh(a.children,t,d,u)),!(a.path==null&&!a.index)&&t.push({path:u,score:n1(u,a.index),routesMeta:d})};return e.forEach((a,o)=>{var s;if(a.path===""||!((s=a.path)!=null&&s.includes("?")))i(a,o);else for(let l of bh(a.path))i(a,o,l)}),t}function bh(e){let t=e.split("/");if(t.length===0)return[];let[n,...r]=t,i=n.endsWith("?"),a=n.replace(/\?$/,"");if(r.length===0)return i?[a,""]:[a];let o=bh(r.join("/")),s=[];return s.push(...o.map(l=>l===""?a:[a,l].join("/"))),i&&s.push(...o),s.map(l=>e.startsWith("/")&&l===""?"/":l)}function X0(e){e.sort((t,n)=>t.score!==n.score?n.score-t.score:r1(t.routesMeta.map(r=>r.childrenIndex),n.routesMeta.map(r=>r.childrenIndex)))}const Y0=/^:[\w-]+$/,G0=3,Z0=2,J0=1,e1=10,t1=-2,Hd=e=>e==="*";function n1(e,t){let n=e.split("/"),r=n.length;return n.some(Hd)&&(r+=t1),t&&(r+=Z0),n.filter(i=>!Hd(i)).reduce((i,a)=>i+(Y0.test(a)?G0:a===""?J0:e1),r)}function r1(e,t){return e.length===t.length&&e.slice(0,-1).every((r,i)=>r===t[i])?e[e.length-1]-t[t.length-1]:0}function i1(e,t,n){let{routesMeta:r}=e,i={},a="/",o=[];for(let s=0;s<r.length;++s){let l=r[s],u=s===r.length-1,d=a==="/"?t:t.slice(a.length)||"/",c=Kl({path:l.relativePath,caseSensitive:l.caseSensitive,end:u},d),p=l.route;if(!c)return null;Object.assign(i,c.params),o.push({params:i,pathname:kn([a,c.pathname]),pathnameBase:d1(kn([a,c.pathnameBase])),route:p}),c.pathnameBase!=="/"&&(a=kn([a,c.pathnameBase]))}return o}function Kl(e,t){typeof e=="string"&&(e={path:e,caseSensitive:!1,end:!0});let[n,r]=a1(e.path,e.caseSensitive,e.end),i=t.match(n);if(!i)return null;let a=i[0],o=a.replace(/(.)\/+$/,"$1"),s=i.slice(1);return{params:r.reduce((u,d,c)=>{let{paramName:p,isOptional:f}=d;if(p==="*"){let v=s[c]||"";o=a.slice(0,a.length-v.length).replace(/(.)\/+$/,"$1")}const y=s[c];return f&&!y?u[p]=void 0:u[p]=(y||"").replace(/%2F/g,"/"),u},{}),pathname:a,pathnameBase:o,pattern:e}}function a1(e,t,n){t===void 0&&(t=!1),n===void 0&&(n=!0),uc(e==="*"||!e.endsWith("*")||e.endsWith("/*"),'Route path "'+e+'" will be treated as if it were '+('"'+e.replace(/\*$/,"/*")+'" because the `*` character must ')+"always follow a `/` in the pattern. To get rid of this warning, "+('please change the route path to "'+e.replace(/\*$/,"/*")+'".'));let r=[],i="^"+e.replace(/\/*\*?$/,"").replace(/^\/*/,"/").replace(/[\\.*+^${}|()[\]]/g,"\\$&").replace(/\/:([\w-]+)(\?)?/g,(o,s,l)=>(r.push({paramName:s,isOptional:l!=null}),l?"/?([^\\/]+)?":"/([^\\/]+)"));return e.endsWith("*")?(r.push({paramName:"*"}),i+=e==="*"||e==="/*"?"(.*)$":"(?:\\/(.+)|\\/*)$"):n?i+="\\/*$":e!==""&&e!=="/"&&(i+="(?:(?=\\/|$))"),[new RegExp(i,t?void 0:"i"),r]}function o1(e){try{return e.split("/").map(t=>decodeURIComponent(t).replace(/\//g,"%2F")).join("/")}catch(t){return uc(!1,'The URL path "'+e+'" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent '+("encoding ("+t+").")),e}}function Dr(e,t){if(t==="/")return e;if(!e.toLowerCase().startsWith(t.toLowerCase()))return null;let n=t.endsWith("/")?t.length-1:t.length,r=e.charAt(n);return r&&r!=="/"?null:e.slice(n)||"/"}const s1=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,l1=e=>s1.test(e);function u1(e,t){t===void 0&&(t="/");let{pathname:n,search:r="",hash:i=""}=typeof e=="string"?Ur(e):e,a;if(n)if(l1(n))a=n;else{if(n.includes("//")){let o=n;n=n.replace(/\/\/+/g,"/"),uc(!1,"Pathnames cannot have embedded double slashes - normalizing "+(o+" -> "+n))}n.startsWith("/")?a=$d(n.substring(1),"/"):a=$d(n,t)}else a=t;return{pathname:a,search:p1(r),hash:f1(i)}}function $d(e,t){let n=t.replace(/\/+$/,"").split("/");return e.split("/").forEach(i=>{i===".."?n.length>1&&n.pop():i!=="."&&n.push(i)}),n.length>1?n.join("/"):"/"}function zs(e,t,n,r){return"Cannot include a '"+e+"' character in a manually specified "+("`to."+t+"` field ["+JSON.stringify(r)+"].  Please separate it out to the ")+("`to."+n+"` field. Alternatively you may provide the full path as ")+'a string in <Link to="..."> and the router will parse it for you.'}function c1(e){return e.filter((t,n)=>n===0||t.route.path&&t.route.path.length>0)}function kh(e,t){let n=c1(e);return t?n.map((r,i)=>i===n.length-1?r.pathname:r.pathnameBase):n.map(r=>r.pathnameBase)}function Sh(e,t,n,r){r===void 0&&(r=!1);let i;typeof e=="string"?i=Ur(e):(i=Ui({},e),we(!i.pathname||!i.pathname.includes("?"),zs("?","pathname","search",i)),we(!i.pathname||!i.pathname.includes("#"),zs("#","pathname","hash",i)),we(!i.search||!i.search.includes("#"),zs("#","search","hash",i)));let a=e===""||i.pathname==="",o=a?"/":i.pathname,s;if(o==null)s=n;else{let c=t.length-1;if(!r&&o.startsWith("..")){let p=o.split("/");for(;p[0]==="..";)p.shift(),c-=1;i.pathname=p.join("/")}s=c>=0?t[c]:"/"}let l=u1(i,s),u=o&&o!=="/"&&o.endsWith("/"),d=(a||o===".")&&n.endsWith("/");return!l.pathname.endsWith("/")&&(u||d)&&(l.pathname+="/"),l}const kn=e=>e.join("/").replace(/\/\/+/g,"/"),d1=e=>e.replace(/\/+$/,"").replace(/^\/*/,"/"),p1=e=>!e||e==="?"?"":e.startsWith("?")?e:"?"+e,f1=e=>!e||e==="#"?"":e.startsWith("#")?e:"#"+e;function m1(e){return e!=null&&typeof e.status=="number"&&typeof e.statusText=="string"&&typeof e.internal=="boolean"&&"data"in e}const Ch=["post","put","patch","delete"];new Set(Ch);const h1=["get",...Ch];new Set(h1);/**
 * React Router v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Vi(){return Vi=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Vi.apply(this,arguments)}const Vo=E.createContext(null),Eh=E.createContext(null),Nn=E.createContext(null),Ho=E.createContext(null),An=E.createContext({outlet:null,matches:[],isDataRoute:!1}),zh=E.createContext(null);function g1(e,t){let{relative:n}=t===void 0?{}:t;ea()||we(!1);let{basename:r,navigator:i}=E.useContext(Nn),{hash:a,pathname:o,search:s}=$o(e,{relative:n}),l=o;return r!=="/"&&(l=o==="/"?r:kn([r,o])),i.createHref({pathname:l,search:s,hash:a})}function ea(){return E.useContext(Ho)!=null}function ta(){return ea()||we(!1),E.useContext(Ho).location}function Th(e){E.useContext(Nn).static||E.useLayoutEffect(e)}function y1(){let{isDataRoute:e}=E.useContext(An);return e?N1():x1()}function x1(){ea()||we(!1);let e=E.useContext(Vo),{basename:t,future:n,navigator:r}=E.useContext(Nn),{matches:i}=E.useContext(An),{pathname:a}=ta(),o=JSON.stringify(kh(i,n.v7_relativeSplatPath)),s=E.useRef(!1);return Th(()=>{s.current=!0}),E.useCallback(function(u,d){if(d===void 0&&(d={}),!s.current)return;if(typeof u=="number"){r.go(u);return}let c=Sh(u,JSON.parse(o),a,d.relative==="path");e==null&&t!=="/"&&(c.pathname=c.pathname==="/"?t:kn([t,c.pathname])),(d.replace?r.replace:r.push)(c,d.state,d)},[t,r,o,a,e])}function Ph(){let{matches:e}=E.useContext(An),t=e[e.length-1];return t?t.params:{}}function $o(e,t){let{relative:n}=t===void 0?{}:t,{future:r}=E.useContext(Nn),{matches:i}=E.useContext(An),{pathname:a}=ta(),o=JSON.stringify(kh(i,r.v7_relativeSplatPath));return E.useMemo(()=>Sh(e,JSON.parse(o),a,n==="path"),[e,o,a,n])}function v1(e,t){return _1(e,t)}function _1(e,t,n,r){ea()||we(!1);let{navigator:i}=E.useContext(Nn),{matches:a}=E.useContext(An),o=a[a.length-1],s=o?o.params:{};o&&o.pathname;let l=o?o.pathnameBase:"/";o&&o.route;let u=ta(),d;if(t){var c;let k=typeof t=="string"?Ur(t):t;l==="/"||(c=k.pathname)!=null&&c.startsWith(l)||we(!1),d=k}else d=u;let p=d.pathname||"/",f=p;if(l!=="/"){let k=l.replace(/^\//,"").split("/");f="/"+p.replace(/^\//,"").split("/").slice(k.length).join("/")}let y=Q0(e,{pathname:f}),v=C1(y&&y.map(k=>Object.assign({},k,{params:Object.assign({},s,k.params),pathname:kn([l,i.encodeLocation?i.encodeLocation(k.pathname).pathname:k.pathname]),pathnameBase:k.pathnameBase==="/"?l:kn([l,i.encodeLocation?i.encodeLocation(k.pathnameBase).pathname:k.pathnameBase])})),a,n,r);return t&&v?E.createElement(Ho.Provider,{value:{location:Vi({pathname:"/",search:"",hash:"",state:null,key:"default"},d),navigationType:mn.Pop}},v):v}function w1(){let e=P1(),t=m1(e)?e.status+" "+e.statusText:e instanceof Error?e.message:JSON.stringify(e),n=e instanceof Error?e.stack:null,i={padding:"0.5rem",backgroundColor:"rgba(200,200,200, 0.5)"};return E.createElement(E.Fragment,null,E.createElement("h2",null,"Unexpected Application Error!"),E.createElement("h3",{style:{fontStyle:"italic"}},t),n?E.createElement("pre",{style:i},n):null,null)}const b1=E.createElement(w1,null);class k1 extends E.Component{constructor(t){super(t),this.state={location:t.location,revalidation:t.revalidation,error:t.error}}static getDerivedStateFromError(t){return{error:t}}static getDerivedStateFromProps(t,n){return n.location!==t.location||n.revalidation!=="idle"&&t.revalidation==="idle"?{error:t.error,location:t.location,revalidation:t.revalidation}:{error:t.error!==void 0?t.error:n.error,location:n.location,revalidation:t.revalidation||n.revalidation}}componentDidCatch(t,n){console.error("React Router caught the following error during render",t,n)}render(){return this.state.error!==void 0?E.createElement(An.Provider,{value:this.props.routeContext},E.createElement(zh.Provider,{value:this.state.error,children:this.props.component})):this.props.children}}function S1(e){let{routeContext:t,match:n,children:r}=e,i=E.useContext(Vo);return i&&i.static&&i.staticContext&&(n.route.errorElement||n.route.ErrorBoundary)&&(i.staticContext._deepestRenderedBoundaryId=n.route.id),E.createElement(An.Provider,{value:t},r)}function C1(e,t,n,r){var i;if(t===void 0&&(t=[]),n===void 0&&(n=null),r===void 0&&(r=null),e==null){var a;if(!n)return null;if(n.errors)e=n.matches;else if((a=r)!=null&&a.v7_partialHydration&&t.length===0&&!n.initialized&&n.matches.length>0)e=n.matches;else return null}let o=e,s=(i=n)==null?void 0:i.errors;if(s!=null){let d=o.findIndex(c=>c.route.id&&(s==null?void 0:s[c.route.id])!==void 0);d>=0||we(!1),o=o.slice(0,Math.min(o.length,d+1))}let l=!1,u=-1;if(n&&r&&r.v7_partialHydration)for(let d=0;d<o.length;d++){let c=o[d];if((c.route.HydrateFallback||c.route.hydrateFallbackElement)&&(u=d),c.route.id){let{loaderData:p,errors:f}=n,y=c.route.loader&&p[c.route.id]===void 0&&(!f||f[c.route.id]===void 0);if(c.route.lazy||y){l=!0,u>=0?o=o.slice(0,u+1):o=[o[0]];break}}}return o.reduceRight((d,c,p)=>{let f,y=!1,v=null,k=null;n&&(f=s&&c.route.id?s[c.route.id]:void 0,v=c.route.errorElement||b1,l&&(u<0&&p===0?(A1("route-fallback"),y=!0,k=null):u===p&&(y=!0,k=c.route.hydrateFallbackElement||null)));let m=t.concat(o.slice(0,p+1)),g=()=>{let x;return f?x=v:y?x=k:c.route.Component?x=E.createElement(c.route.Component,null):c.route.element?x=c.route.element:x=d,E.createElement(S1,{match:c,routeContext:{outlet:d,matches:m,isDataRoute:n!=null},children:x})};return n&&(c.route.ErrorBoundary||c.route.errorElement||p===0)?E.createElement(k1,{location:n.location,revalidation:n.revalidation,component:v,error:f,children:g(),routeContext:{outlet:null,matches:m,isDataRoute:!0}}):g()},null)}var Nh=function(e){return e.UseBlocker="useBlocker",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e}(Nh||{}),Ah=function(e){return e.UseBlocker="useBlocker",e.UseLoaderData="useLoaderData",e.UseActionData="useActionData",e.UseRouteError="useRouteError",e.UseNavigation="useNavigation",e.UseRouteLoaderData="useRouteLoaderData",e.UseMatches="useMatches",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e.UseRouteId="useRouteId",e}(Ah||{});function E1(e){let t=E.useContext(Vo);return t||we(!1),t}function z1(e){let t=E.useContext(Eh);return t||we(!1),t}function T1(e){let t=E.useContext(An);return t||we(!1),t}function Lh(e){let t=T1(),n=t.matches[t.matches.length-1];return n.route.id||we(!1),n.route.id}function P1(){var e;let t=E.useContext(zh),n=z1(),r=Lh();return t!==void 0?t:(e=n.errors)==null?void 0:e[r]}function N1(){let{router:e}=E1(Nh.UseNavigateStable),t=Lh(Ah.UseNavigateStable),n=E.useRef(!1);return Th(()=>{n.current=!0}),E.useCallback(function(i,a){a===void 0&&(a={}),n.current&&(typeof i=="number"?e.navigate(i):e.navigate(i,Vi({fromRouteId:t},a)))},[e,t])}const Qd={};function A1(e,t,n){Qd[e]||(Qd[e]=!0)}function L1(e,t){e==null||e.v7_startTransition,e==null||e.v7_relativeSplatPath}function di(e){we(!1)}function j1(e){let{basename:t="/",children:n=null,location:r,navigationType:i=mn.Pop,navigator:a,static:o=!1,future:s}=e;ea()&&we(!1);let l=t.replace(/^\/*/,"/"),u=E.useMemo(()=>({basename:l,navigator:a,static:o,future:Vi({v7_relativeSplatPath:!1},s)}),[l,s,a,o]);typeof r=="string"&&(r=Ur(r));let{pathname:d="/",search:c="",hash:p="",state:f=null,key:y="default"}=r,v=E.useMemo(()=>{let k=Dr(d,l);return k==null?null:{location:{pathname:k,search:c,hash:p,state:f,key:y},navigationType:i}},[l,d,c,p,f,y,i]);return v==null?null:E.createElement(Nn.Provider,{value:u},E.createElement(Ho.Provider,{children:n,value:v}))}function R1(e){let{children:t,location:n}=e;return v1(Xl(t),n)}new Promise(()=>{});function Xl(e,t){t===void 0&&(t=[]);let n=[];return E.Children.forEach(e,(r,i)=>{if(!E.isValidElement(r))return;let a=[...t,i];if(r.type===E.Fragment){n.push.apply(n,Xl(r.props.children,a));return}r.type!==di&&we(!1),!r.props.index||!r.props.children||we(!1);let o={id:r.props.id||a.join("-"),caseSensitive:r.props.caseSensitive,element:r.props.element,Component:r.props.Component,index:r.props.index,path:r.props.path,loader:r.props.loader,action:r.props.action,errorElement:r.props.errorElement,ErrorBoundary:r.props.ErrorBoundary,hasErrorBoundary:r.props.ErrorBoundary!=null||r.props.errorElement!=null,shouldRevalidate:r.props.shouldRevalidate,handle:r.props.handle,lazy:r.props.lazy};r.props.children&&(o.children=Xl(r.props.children,a)),n.push(o)}),n}/**
 * React Router DOM v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function bo(){return bo=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},bo.apply(this,arguments)}function jh(e,t){if(e==null)return{};var n={},r=Object.keys(e),i,a;for(a=0;a<r.length;a++)i=r[a],!(t.indexOf(i)>=0)&&(n[i]=e[i]);return n}function I1(e){return!!(e.metaKey||e.altKey||e.ctrlKey||e.shiftKey)}function O1(e,t){return e.button===0&&(!t||t==="_self")&&!I1(e)}const D1=["onClick","relative","reloadDocument","replace","state","target","to","preventScrollReset","viewTransition"],M1=["aria-current","caseSensitive","className","end","style","to","viewTransition","children"],F1="6";try{window.__reactRouterVersion=F1}catch{}const q1=E.createContext({isTransitioning:!1}),W1="startTransition",Kd=Ly[W1];function B1(e){let{basename:t,children:n,future:r,window:i}=e,a=E.useRef();a.current==null&&(a.current=V0({window:i,v5Compat:!0}));let o=a.current,[s,l]=E.useState({action:o.action,location:o.location}),{v7_startTransition:u}=r||{},d=E.useCallback(c=>{u&&Kd?Kd(()=>l(c)):l(c)},[l,u]);return E.useLayoutEffect(()=>o.listen(d),[o,d]),E.useEffect(()=>L1(r),[r]),E.createElement(j1,{basename:t,children:n,location:s.location,navigationType:s.action,navigator:o,future:r})}const U1=typeof window<"u"&&typeof window.document<"u"&&typeof window.document.createElement<"u",V1=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,Nt=E.forwardRef(function(t,n){let{onClick:r,relative:i,reloadDocument:a,replace:o,state:s,target:l,to:u,preventScrollReset:d,viewTransition:c}=t,p=jh(t,D1),{basename:f}=E.useContext(Nn),y,v=!1;if(typeof u=="string"&&V1.test(u)&&(y=u,U1))try{let x=new URL(window.location.href),C=u.startsWith("//")?new URL(x.protocol+u):new URL(u),z=Dr(C.pathname,f);C.origin===x.origin&&z!=null?u=z+C.search+C.hash:v=!0}catch{}let k=g1(u,{relative:i}),m=Q1(u,{replace:o,state:s,target:l,preventScrollReset:d,relative:i,viewTransition:c});function g(x){r&&r(x),x.defaultPrevented||m(x)}return E.createElement("a",bo({},p,{href:y||k,onClick:v||a?r:g,ref:n,target:l}))}),H1=E.forwardRef(function(t,n){let{"aria-current":r="page",caseSensitive:i=!1,className:a="",end:o=!1,style:s,to:l,viewTransition:u,children:d}=t,c=jh(t,M1),p=$o(l,{relative:c.relative}),f=ta(),y=E.useContext(Eh),{navigator:v,basename:k}=E.useContext(Nn),m=y!=null&&K1(p)&&u===!0,g=v.encodeLocation?v.encodeLocation(p).pathname:p.pathname,x=f.pathname,C=y&&y.navigation&&y.navigation.location?y.navigation.location.pathname:null;i||(x=x.toLowerCase(),C=C?C.toLowerCase():null,g=g.toLowerCase()),C&&k&&(C=Dr(C,k)||C);const z=g!=="/"&&g.endsWith("/")?g.length-1:g.length;let S=x===g||!o&&x.startsWith(g)&&x.charAt(z)==="/",T=C!=null&&(C===g||!o&&C.startsWith(g)&&C.charAt(g.length)==="/"),N={isActive:S,isPending:T,isTransitioning:m},F=S?r:void 0,b;typeof a=="function"?b=a(N):b=[a,S?"active":null,T?"pending":null,m?"transitioning":null].filter(Boolean).join(" ");let I=typeof s=="function"?s(N):s;return E.createElement(Nt,bo({},c,{"aria-current":F,className:b,ref:n,style:I,to:l,viewTransition:u}),typeof d=="function"?d(N):d)});var Yl;(function(e){e.UseScrollRestoration="useScrollRestoration",e.UseSubmit="useSubmit",e.UseSubmitFetcher="useSubmitFetcher",e.UseFetcher="useFetcher",e.useViewTransitionState="useViewTransitionState"})(Yl||(Yl={}));var Xd;(function(e){e.UseFetcher="useFetcher",e.UseFetchers="useFetchers",e.UseScrollRestoration="useScrollRestoration"})(Xd||(Xd={}));function $1(e){let t=E.useContext(Vo);return t||we(!1),t}function Q1(e,t){let{target:n,replace:r,state:i,preventScrollReset:a,relative:o,viewTransition:s}=t===void 0?{}:t,l=y1(),u=ta(),d=$o(e,{relative:o});return E.useCallback(c=>{if(O1(c,n)){c.preventDefault();let p=r!==void 0?r:wo(u)===wo(d);l(e,{replace:p,state:i,preventScrollReset:a,relative:o,viewTransition:s})}},[u,l,d,r,i,n,e,a,o,s])}function K1(e,t){t===void 0&&(t={});let n=E.useContext(q1);n==null&&we(!1);let{basename:r}=$1(Yl.useViewTransitionState),i=$o(e,{relative:t.relative});if(!n.isTransitioning)return!1;let a=Dr(n.currentLocation.pathname,r)||n.currentLocation.pathname,o=Dr(n.nextLocation.pathname,r)||n.nextLocation.pathname;return Kl(i.pathname,o)!=null||Kl(i.pathname,a)!=null}var X1=typeof Element<"u",Y1=typeof Map=="function",G1=typeof Set=="function",Z1=typeof ArrayBuffer=="function"&&!!ArrayBuffer.isView;function Ha(e,t){if(e===t)return!0;if(e&&t&&typeof e=="object"&&typeof t=="object"){if(e.constructor!==t.constructor)return!1;var n,r,i;if(Array.isArray(e)){if(n=e.length,n!=t.length)return!1;for(r=n;r--!==0;)if(!Ha(e[r],t[r]))return!1;return!0}var a;if(Y1&&e instanceof Map&&t instanceof Map){if(e.size!==t.size)return!1;for(a=e.entries();!(r=a.next()).done;)if(!t.has(r.value[0]))return!1;for(a=e.entries();!(r=a.next()).done;)if(!Ha(r.value[1],t.get(r.value[0])))return!1;return!0}if(G1&&e instanceof Set&&t instanceof Set){if(e.size!==t.size)return!1;for(a=e.entries();!(r=a.next()).done;)if(!t.has(r.value[0]))return!1;return!0}if(Z1&&ArrayBuffer.isView(e)&&ArrayBuffer.isView(t)){if(n=e.length,n!=t.length)return!1;for(r=n;r--!==0;)if(e[r]!==t[r])return!1;return!0}if(e.constructor===RegExp)return e.source===t.source&&e.flags===t.flags;if(e.valueOf!==Object.prototype.valueOf&&typeof e.valueOf=="function"&&typeof t.valueOf=="function")return e.valueOf()===t.valueOf();if(e.toString!==Object.prototype.toString&&typeof e.toString=="function"&&typeof t.toString=="function")return e.toString()===t.toString();if(i=Object.keys(e),n=i.length,n!==Object.keys(t).length)return!1;for(r=n;r--!==0;)if(!Object.prototype.hasOwnProperty.call(t,i[r]))return!1;if(X1&&e instanceof Element)return!1;for(r=n;r--!==0;)if(!((i[r]==="_owner"||i[r]==="__v"||i[r]==="__o")&&e.$$typeof)&&!Ha(e[i[r]],t[i[r]]))return!1;return!0}return e!==e&&t!==t}var J1=function(t,n){try{return Ha(t,n)}catch(r){if((r.message||"").match(/stack|recursion/i))return console.warn("react-fast-compare cannot handle circular refs"),!1;throw r}};const ev=Zn(J1);var tv=function(e,t,n,r,i,a,o,s){if(!e){var l;if(t===void 0)l=new Error("Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.");else{var u=[n,r,i,a,o,s],d=0;l=new Error(t.replace(/%s/g,function(){return u[d++]})),l.name="Invariant Violation"}throw l.framesToPop=1,l}},nv=tv;const Yd=Zn(nv);var rv=function(t,n,r,i){var a=r?r.call(i,t,n):void 0;if(a!==void 0)return!!a;if(t===n)return!0;if(typeof t!="object"||!t||typeof n!="object"||!n)return!1;var o=Object.keys(t),s=Object.keys(n);if(o.length!==s.length)return!1;for(var l=Object.prototype.hasOwnProperty.bind(n),u=0;u<o.length;u++){var d=o[u];if(!l(d))return!1;var c=t[d],p=n[d];if(a=r?r.call(i,c,p,d):void 0,a===!1||a===void 0&&c!==p)return!1}return!0};const iv=Zn(rv);var Rh=(e=>(e.BASE="base",e.BODY="body",e.HEAD="head",e.HTML="html",e.LINK="link",e.META="meta",e.NOSCRIPT="noscript",e.SCRIPT="script",e.STYLE="style",e.TITLE="title",e.FRAGMENT="Symbol(react.fragment)",e))(Rh||{}),Ts={link:{rel:["amphtml","canonical","alternate"]},script:{type:["application/ld+json"]},meta:{charset:"",name:["generator","robots","description"],property:["og:type","og:title","og:url","og:image","og:image:alt","og:description","twitter:url","twitter:title","twitter:description","twitter:image","twitter:image:alt","twitter:card","twitter:site"]}},Gd=Object.values(Rh),cc={accesskey:"accessKey",charset:"charSet",class:"className",contenteditable:"contentEditable",contextmenu:"contextMenu","http-equiv":"httpEquiv",itemprop:"itemProp",tabindex:"tabIndex"},av=Object.entries(cc).reduce((e,[t,n])=>(e[n]=t,e),{}),At="data-rh",Er={DEFAULT_TITLE:"defaultTitle",DEFER:"defer",ENCODE_SPECIAL_CHARACTERS:"encodeSpecialCharacters",ON_CHANGE_CLIENT_STATE:"onChangeClientState",TITLE_TEMPLATE:"titleTemplate",PRIORITIZE_SEO_TAGS:"prioritizeSeoTags"},zr=(e,t)=>{for(let n=e.length-1;n>=0;n-=1){const r=e[n];if(Object.prototype.hasOwnProperty.call(r,t))return r[t]}return null},ov=e=>{let t=zr(e,"title");const n=zr(e,Er.TITLE_TEMPLATE);if(Array.isArray(t)&&(t=t.join("")),n&&t)return n.replace(/%s/g,()=>t);const r=zr(e,Er.DEFAULT_TITLE);return t||r||void 0},sv=e=>zr(e,Er.ON_CHANGE_CLIENT_STATE)||(()=>{}),Ps=(e,t)=>t.filter(n=>typeof n[e]<"u").map(n=>n[e]).reduce((n,r)=>({...n,...r}),{}),lv=(e,t)=>t.filter(n=>typeof n.base<"u").map(n=>n.base).reverse().reduce((n,r)=>{if(!n.length){const i=Object.keys(r);for(let a=0;a<i.length;a+=1){const s=i[a].toLowerCase();if(e.indexOf(s)!==-1&&r[s])return n.concat(r)}}return n},[]),uv=e=>console&&typeof console.warn=="function"&&console.warn(e),ni=(e,t,n)=>{const r={};return n.filter(i=>Array.isArray(i[e])?!0:(typeof i[e]<"u"&&uv(`Helmet: ${e} should be of type "Array". Instead found type "${typeof i[e]}"`),!1)).map(i=>i[e]).reverse().reduce((i,a)=>{const o={};a.filter(l=>{let u;const d=Object.keys(l);for(let p=0;p<d.length;p+=1){const f=d[p],y=f.toLowerCase();t.indexOf(y)!==-1&&!(u==="rel"&&l[u].toLowerCase()==="canonical")&&!(y==="rel"&&l[y].toLowerCase()==="stylesheet")&&(u=y),t.indexOf(f)!==-1&&(f==="innerHTML"||f==="cssText"||f==="itemprop")&&(u=f)}if(!u||!l[u])return!1;const c=l[u].toLowerCase();return r[u]||(r[u]={}),o[u]||(o[u]={}),r[u][c]?!1:(o[u][c]=!0,!0)}).reverse().forEach(l=>i.push(l));const s=Object.keys(o);for(let l=0;l<s.length;l+=1){const u=s[l],d={...r[u],...o[u]};r[u]=d}return i},[]).reverse()},cv=(e,t)=>{if(Array.isArray(e)&&e.length){for(let n=0;n<e.length;n+=1)if(e[n][t])return!0}return!1},dv=e=>({baseTag:lv(["href"],e),bodyAttributes:Ps("bodyAttributes",e),defer:zr(e,Er.DEFER),encode:zr(e,Er.ENCODE_SPECIAL_CHARACTERS),htmlAttributes:Ps("htmlAttributes",e),linkTags:ni("link",["rel","href"],e),metaTags:ni("meta",["name","charset","http-equiv","property","itemprop"],e),noscriptTags:ni("noscript",["innerHTML"],e),onChangeClientState:sv(e),scriptTags:ni("script",["src","innerHTML"],e),styleTags:ni("style",["cssText"],e),title:ov(e),titleAttributes:Ps("titleAttributes",e),prioritizeSeoTags:cv(e,Er.PRIORITIZE_SEO_TAGS)}),Ih=e=>Array.isArray(e)?e.join(""):e,pv=(e,t)=>{const n=Object.keys(e);for(let r=0;r<n.length;r+=1)if(t[n[r]]&&t[n[r]].includes(e[n[r]]))return!0;return!1},Ns=(e,t)=>Array.isArray(e)?e.reduce((n,r)=>(pv(r,t)?n.priority.push(r):n.default.push(r),n),{priority:[],default:[]}):{default:e,priority:[]},Zd=(e,t)=>({...e,[t]:void 0}),fv=["noscript","script","style"],Gl=(e,t=!0)=>t===!1?String(e):String(e).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#x27;"),Oh=e=>Object.keys(e).reduce((t,n)=>{const r=typeof e[n]<"u"?`${n}="${e[n]}"`:`${n}`;return t?`${t} ${r}`:r},""),mv=(e,t,n,r)=>{const i=Oh(n),a=Ih(t);return i?`<${e} ${At}="true" ${i}>${Gl(a,r)}</${e}>`:`<${e} ${At}="true">${Gl(a,r)}</${e}>`},hv=(e,t,n=!0)=>t.reduce((r,i)=>{const a=i,o=Object.keys(a).filter(u=>!(u==="innerHTML"||u==="cssText")).reduce((u,d)=>{const c=typeof a[d]>"u"?d:`${d}="${Gl(a[d],n)}"`;return u?`${u} ${c}`:c},""),s=a.innerHTML||a.cssText||"",l=fv.indexOf(e)===-1;return`${r}<${e} ${At}="true" ${o}${l?"/>":`>${s}</${e}>`}`},""),Dh=(e,t={})=>Object.keys(e).reduce((n,r)=>{const i=cc[r];return n[i||r]=e[r],n},t),gv=(e,t,n)=>{const r={key:t,[At]:!0},i=Dh(n,r);return[Ne.createElement("title",i,t)]},$a=(e,t)=>t.map((n,r)=>{const i={key:r,[At]:!0};return Object.keys(n).forEach(a=>{const s=cc[a]||a;if(s==="innerHTML"||s==="cssText"){const l=n.innerHTML||n.cssText;i.dangerouslySetInnerHTML={__html:l}}else i[s]=n[a]}),Ne.createElement(e,i)}),ft=(e,t,n=!0)=>{switch(e){case"title":return{toComponent:()=>gv(e,t.title,t.titleAttributes),toString:()=>mv(e,t.title,t.titleAttributes,n)};case"bodyAttributes":case"htmlAttributes":return{toComponent:()=>Dh(t),toString:()=>Oh(t)};default:return{toComponent:()=>$a(e,t),toString:()=>hv(e,t,n)}}},yv=({metaTags:e,linkTags:t,scriptTags:n,encode:r})=>{const i=Ns(e,Ts.meta),a=Ns(t,Ts.link),o=Ns(n,Ts.script);return{priorityMethods:{toComponent:()=>[...$a("meta",i.priority),...$a("link",a.priority),...$a("script",o.priority)],toString:()=>`${ft("meta",i.priority,r)} ${ft("link",a.priority,r)} ${ft("script",o.priority,r)}`},metaTags:i.default,linkTags:a.default,scriptTags:o.default}},xv=e=>{const{baseTag:t,bodyAttributes:n,encode:r=!0,htmlAttributes:i,noscriptTags:a,styleTags:o,title:s="",titleAttributes:l,prioritizeSeoTags:u}=e;let{linkTags:d,metaTags:c,scriptTags:p}=e,f={toComponent:()=>{},toString:()=>""};return u&&({priorityMethods:f,linkTags:d,metaTags:c,scriptTags:p}=yv(e)),{priority:f,base:ft("base",t,r),bodyAttributes:ft("bodyAttributes",n,r),htmlAttributes:ft("htmlAttributes",i,r),link:ft("link",d,r),meta:ft("meta",c,r),noscript:ft("noscript",a,r),script:ft("script",p,r),style:ft("style",o,r),title:ft("title",{title:s,titleAttributes:l},r)}},Zl=xv,Sa=[],Mh=!!(typeof window<"u"&&window.document&&window.document.createElement),Jl=class{constructor(e,t){$t(this,"instances",[]);$t(this,"canUseDOM",Mh);$t(this,"context");$t(this,"value",{setHelmet:e=>{this.context.helmet=e},helmetInstances:{get:()=>this.canUseDOM?Sa:this.instances,add:e=>{(this.canUseDOM?Sa:this.instances).push(e)},remove:e=>{const t=(this.canUseDOM?Sa:this.instances).indexOf(e);(this.canUseDOM?Sa:this.instances).splice(t,1)}}});this.context=e,this.canUseDOM=t||!1,t||(e.helmet=Zl({baseTag:[],bodyAttributes:{},htmlAttributes:{},linkTags:[],metaTags:[],noscriptTags:[],scriptTags:[],styleTags:[],title:"",titleAttributes:{}}))}},vv={},Fh=Ne.createContext(vv),Un,qh=(Un=class extends E.Component{constructor(n){super(n);$t(this,"helmetData");this.helmetData=new Jl(this.props.context||{},Un.canUseDOM)}render(){return Ne.createElement(Fh.Provider,{value:this.helmetData.value},this.props.children)}},$t(Un,"canUseDOM",Mh),Un),ir=(e,t)=>{const n=document.head||document.querySelector("head"),r=n.querySelectorAll(`${e}[${At}]`),i=[].slice.call(r),a=[];let o;return t&&t.length&&t.forEach(s=>{const l=document.createElement(e);for(const u in s)if(Object.prototype.hasOwnProperty.call(s,u))if(u==="innerHTML")l.innerHTML=s.innerHTML;else if(u==="cssText")l.styleSheet?l.styleSheet.cssText=s.cssText:l.appendChild(document.createTextNode(s.cssText));else{const d=u,c=typeof s[d]>"u"?"":s[d];l.setAttribute(u,c)}l.setAttribute(At,"true"),i.some((u,d)=>(o=d,l.isEqualNode(u)))?i.splice(o,1):a.push(l)}),i.forEach(s=>{var l;return(l=s.parentNode)==null?void 0:l.removeChild(s)}),a.forEach(s=>n.appendChild(s)),{oldTags:i,newTags:a}},eu=(e,t)=>{const n=document.getElementsByTagName(e)[0];if(!n)return;const r=n.getAttribute(At),i=r?r.split(","):[],a=[...i],o=Object.keys(t);for(const s of o){const l=t[s]||"";n.getAttribute(s)!==l&&n.setAttribute(s,l),i.indexOf(s)===-1&&i.push(s);const u=a.indexOf(s);u!==-1&&a.splice(u,1)}for(let s=a.length-1;s>=0;s-=1)n.removeAttribute(a[s]);i.length===a.length?n.removeAttribute(At):n.getAttribute(At)!==o.join(",")&&n.setAttribute(At,o.join(","))},_v=(e,t)=>{typeof e<"u"&&document.title!==e&&(document.title=Ih(e)),eu("title",t)},Jd=(e,t)=>{const{baseTag:n,bodyAttributes:r,htmlAttributes:i,linkTags:a,metaTags:o,noscriptTags:s,onChangeClientState:l,scriptTags:u,styleTags:d,title:c,titleAttributes:p}=e;eu("body",r),eu("html",i),_v(c,p);const f={baseTag:ir("base",n),linkTags:ir("link",a),metaTags:ir("meta",o),noscriptTags:ir("noscript",s),scriptTags:ir("script",u),styleTags:ir("style",d)},y={},v={};Object.keys(f).forEach(k=>{const{newTags:m,oldTags:g}=f[k];m.length&&(y[k]=m),g.length&&(v[k]=f[k].oldTags)}),t&&t(),l(e,y,v)},ri=null,wv=e=>{ri&&cancelAnimationFrame(ri),e.defer?ri=requestAnimationFrame(()=>{Jd(e,()=>{ri=null})}):(Jd(e),ri=null)},bv=wv,ep=class extends E.Component{constructor(){super(...arguments);$t(this,"rendered",!1)}shouldComponentUpdate(t){return!iv(t,this.props)}componentDidUpdate(){this.emitChange()}componentWillUnmount(){const{helmetInstances:t}=this.props.context;t.remove(this),this.emitChange()}emitChange(){const{helmetInstances:t,setHelmet:n}=this.props.context;let r=null;const i=dv(t.get().map(a=>{const o={...a.props};return delete o.context,o}));qh.canUseDOM?bv(i):Zl&&(r=Zl(i)),n(r)}init(){if(this.rendered)return;this.rendered=!0;const{helmetInstances:t}=this.props.context;t.add(this),this.emitChange()}render(){return this.init(),null}},Js,kv=(Js=class extends E.Component{shouldComponentUpdate(e){return!ev(Zd(this.props,"helmetData"),Zd(e,"helmetData"))}mapNestedChildrenToProps(e,t){if(!t)return null;switch(e.type){case"script":case"noscript":return{innerHTML:t};case"style":return{cssText:t};default:throw new Error(`<${e.type} /> elements are self-closing and can not contain children. Refer to our API for more information.`)}}flattenArrayTypeChildren(e,t,n,r){return{...t,[e.type]:[...t[e.type]||[],{...n,...this.mapNestedChildrenToProps(e,r)}]}}mapObjectTypeChildren(e,t,n,r){switch(e.type){case"title":return{...t,[e.type]:r,titleAttributes:{...n}};case"body":return{...t,bodyAttributes:{...n}};case"html":return{...t,htmlAttributes:{...n}};default:return{...t,[e.type]:{...n}}}}mapArrayTypeChildrenToProps(e,t){let n={...t};return Object.keys(e).forEach(r=>{n={...n,[r]:e[r]}}),n}warnOnInvalidChildren(e,t){return Yd(Gd.some(n=>e.type===n),typeof e.type=="function"?"You may be attempting to nest <Helmet> components within each other, which is not allowed. Refer to our API for more information.":`Only elements types ${Gd.join(", ")} are allowed. Helmet does not support rendering <${e.type}> elements. Refer to our API for more information.`),Yd(!t||typeof t=="string"||Array.isArray(t)&&!t.some(n=>typeof n!="string"),`Helmet expects a string as a child of <${e.type}>. Did you forget to wrap your children in braces? ( <${e.type}>{\`\`}</${e.type}> ) Refer to our API for more information.`),!0}mapChildrenToProps(e,t){let n={};return Ne.Children.forEach(e,r=>{if(!r||!r.props)return;const{children:i,...a}=r.props,o=Object.keys(a).reduce((l,u)=>(l[av[u]||u]=a[u],l),{});let{type:s}=r;switch(typeof s=="symbol"?s=s.toString():this.warnOnInvalidChildren(r,i),s){case"Symbol(react.fragment)":t=this.mapChildrenToProps(i,t);break;case"link":case"meta":case"noscript":case"script":case"style":n=this.flattenArrayTypeChildren(r,n,o,i);break;default:t=this.mapObjectTypeChildren(r,t,o,i);break}}),this.mapArrayTypeChildrenToProps(n,t)}render(){const{children:e,...t}=this.props;let n={...t},{helmetData:r}=t;if(e&&(n=this.mapChildrenToProps(e,n)),r&&!(r instanceof Jl)){const i=r;r=new Jl(i.context,!0),delete n.helmetData}return r?Ne.createElement(ep,{...n,context:r.value}):Ne.createElement(Fh.Consumer,null,i=>Ne.createElement(ep,{...n,context:i}))}},$t(Js,"defaultProps",{defer:!0,encodeSpecialCharacters:!0,prioritizeSeoTags:!1}),Js);const As="ml-interview-progress",Wh=E.createContext(null);function Sv({children:e}){const[t,n]=E.useState(()=>{const u=localStorage.getItem(As);return u?JSON.parse(u):{}});E.useEffect(()=>{localStorage.setItem(As,JSON.stringify(t))},[t]);const r=E.useCallback((u,d,c)=>{n(p=>{var f;return{...p,[u]:{...p[u],[d]:{...(f=p[u])==null?void 0:f[d],status:c,lastAttempt:new Date().toISOString()}}}})},[]),i=E.useCallback((u,d,c)=>{n(p=>{var f,y,v;return{...p,[u]:{...p[u],[d]:{...(f=p[u])==null?void 0:f[d],status:((v=(y=p[u])==null?void 0:y[d])==null?void 0:v.status)||"in_progress",code:c,lastAttempt:new Date().toISOString()}}}})},[]),a=E.useCallback((u,d)=>{var p;const c=(p=t[u])==null?void 0:p[d];return{status:(c==null?void 0:c.status)||"not_started",code:c==null?void 0:c.code}},[t]),o=E.useCallback((u,d)=>{const c=t[u];if(!c||d===0)return 0;const p=Object.values(c).filter(f=>f.status==="completed").length;return Math.round(p/d*100)},[t]),s=E.useCallback(u=>{const d=u.reduce((p,f)=>p+f.problemCount,0);if(d===0)return 0;let c=0;for(const p of u){const f=t[p.id];f&&(c+=Object.values(f).filter(y=>y.status==="completed").length)}return Math.round(c/d*100)},[t]),l=E.useCallback(()=>{n({}),localStorage.removeItem(As)},[]);return h.jsx(Wh.Provider,{value:{progress:t,updateProblemStatus:r,saveProblemCode:i,getProblemProgress:a,getSectionProgress:o,getOverallProgress:s,resetProgress:l},children:e})}function na(){const e=E.useContext(Wh);if(!e)throw new Error("useProgress must be used within a ProgressProvider");return e}const Mr=[{id:"numpy-fundamentals",title:"NumPy Fundamentals",description:"Master NumPy array operations essential for ML development.",icon:"🔢",introduction:"\n# NumPy Fundamentals\n\nNumPy is the backbone of scientific computing in Python. Every ML library builds on it.\n\n## Key Concepts\n\n### Array Creation\n- `np.array()`: Create from lists\n- `np.zeros()`, `np.ones()`: Initialize with values\n- `np.arange()`, `np.linspace()`: Numeric ranges\n- `np.eye()`: Identity matrix\n\n### Indexing & Slicing\n- Basic: `arr[0]`, `arr[1:5]`\n- Advanced: `arr[[0, 2, 4]]`, `arr[arr > 0]`\n- Multi-dimensional: `arr[1, :]`, `arr[:, 2:4]`\n\n### Broadcasting\nArrays of different shapes can be combined:\n- (3, 4) + (4,) → broadcasts to (3, 4)\n- (3, 1) * (1, 4) → broadcasts to (3, 4)\n\n### Aggregations\n- `sum()`, `mean()`, `std()` - with axis parameter\n- `argmax()`, `argmin()` - find indices\n\n### Shape Manipulation\n- `reshape()`: Change dimensions\n- `transpose()`, `.T`: Swap axes\n- `flatten()`, `ravel()`: To 1D\n\nMaster these operations to write efficient ML code!\n    ",problems:["numpy-array-creation","numpy-indexing","numpy-broadcasting","numpy-aggregations","numpy-reshape-transpose"]},{id:"python-basics",title:"Python Basics for ML",description:"Master NumPy arrays and essential Python operations for machine learning.",icon:"🐍",introduction:"\n# Python Basics for Machine Learning\n\nNumPy is the foundation of nearly all machine learning in Python. Before diving into complex algorithms, you need to master array operations.\n\n## Key Concepts\n\n### NumPy Arrays\n- **Creation**: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`\n- **Shape manipulation**: `reshape()`, `flatten()`, `transpose()`\n- **Indexing**: Slicing, boolean indexing, fancy indexing\n\n### Essential Operations\n- **Element-wise**: Addition, multiplication, mathematical functions\n- **Aggregations**: `sum()`, `mean()`, `std()`, `min()`, `max()`\n- **Broadcasting**: Automatic expansion of arrays for operations\n\n### Why It Matters\nEvery ML algorithm processes data as arrays. Understanding these operations lets you:\n- Prepare data efficiently\n- Debug model issues\n- Optimize performance\n\nLet's practice with hands-on problems!\n    ",problems:["numpy-array-sum","numpy-matrix-multiply","numpy-broadcast-add"]},{id:"einsum",title:"Einstein Summation (Einsum)",description:"Master einsum for elegant tensor operations.",icon:"∑",introduction:`
# Einstein Summation (Einsum)

Einsum is a powerful notation for expressing tensor operations concisely.

## Why Einsum?
- Express complex operations in one line
- Often faster than chained operations
- Essential for attention mechanisms

## Syntax
\`\`\`python
np.einsum('subscripts', operands)
\`\`\`

## Common Patterns

### Basic Operations
\`\`\`
'ij->'      Sum all elements
'ij->i'     Sum along axis 1 (row sums)
'ij->j'     Sum along axis 0 (column sums)
'ij->ji'    Transpose
'ii->i'     Extract diagonal
'ii->'      Trace (sum of diagonal)
\`\`\`

### Matrix Operations
\`\`\`
'ik,kj->ij'   Matrix multiplication (A @ B)
'ij,ij->ij'   Element-wise product (A * B)
'ij,ij->'     Frobenius inner product
'i,j->ij'     Outer product
'i,i->'       Dot product
\`\`\`

### Batch Operations
\`\`\`
'bij,bjk->bik'   Batch matrix multiply
'bqd,bkd->bqk'   Attention scores (Q @ K.T)
'bhqk,bhkd->bhqd' Multi-head attention output
\`\`\`

## Key Insight
Letters that appear on both sides are kept.
Letters that disappear are summed over (contracted).

Master einsum to write clean, efficient deep learning code!
    `,problems:["einsum-basics","einsum-matrix-ops","einsum-batch-ops","einsum-advanced","einsum-vs-matmul"]},{id:"pytorch-basics",title:"PyTorch Basics",description:"Learn PyTorch patterns implemented in NumPy.",icon:"🔥",introduction:`
# PyTorch Basics

Learn PyTorch patterns and concepts. Since Pyodide doesn't support PyTorch, we implement these concepts in NumPy.

## Tensor Operations

### Creation
\`\`\`python
# PyTorch           # NumPy equivalent
torch.tensor()      np.array()
torch.zeros()       np.zeros()
torch.randn()       np.random.randn()
\`\`\`

### Operations
\`\`\`python
# PyTorch           # NumPy equivalent
x + y               x + y
x @ y               x @ y
x.sum(dim=1)        x.sum(axis=1)
x.view(2, -1)       x.reshape(2, -1)
\`\`\`

## Autograd Concepts

PyTorch tracks operations for automatic differentiation:
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # dy/dx = 2x = 4
\`\`\`

We implement this manually to understand the math.

## Module Pattern
\`\`\`python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight = Parameter(...)
        self.bias = Parameter(...)

    def forward(self, x):
        return x @ self.weight + self.bias
\`\`\`

## Loss Functions
- CrossEntropyLoss: Classification
- MSELoss: Regression
- BCELoss: Binary classification

Learn these patterns to understand PyTorch!
    `,problems:["tensor-creation","tensor-operations","autograd-concepts","nn-modules","loss-functions"]},{id:"data-preprocessing",title:"Data Preprocessing",description:"Learn essential data cleaning and transformation techniques.",icon:"🧹",introduction:`
# Data Preprocessing

Real-world data is messy. Before training any model, you need to clean and transform your data.

## Key Concepts

### Handling Missing Data
- **Detection**: \`np.isnan()\`, \`pd.isnull()\`
- **Strategies**: Remove, fill with mean/median/mode, interpolate

### Feature Scaling
- **Normalization**: Scale to [0, 1] range using min-max scaling
- **Standardization**: Transform to zero mean, unit variance (z-score)
- **When to use**: Most algorithms need scaled features

### Encoding Categorical Data
- **One-Hot Encoding**: Convert categories to binary vectors
- **Label Encoding**: Map categories to integers
- **When to use**: Tree-based models handle label encoding; others need one-hot

### Why It Matters
Garbage in, garbage out. Proper preprocessing can:
- Improve model accuracy by 10-30%
- Prevent training failures
- Reduce overfitting

Let's practice these essential skills!
    `,problems:["normalize-features","handle-missing-data","one-hot-encode"]},{id:"supervised-learning",title:"Supervised Learning",description:"Implement core supervised learning algorithms from scratch.",icon:"📊",introduction:`
# Supervised Learning

Supervised learning is when you train a model on labeled data to make predictions.

## Key Concepts

### Linear Regression
- **Goal**: Predict continuous values
- **Formula**: y = wx + b
- **Training**: Minimize mean squared error (MSE)
- **Gradient descent**: Update weights iteratively

### Logistic Regression
- **Goal**: Binary classification
- **Sigmoid function**: Maps outputs to [0, 1] probabilities
- **Loss**: Binary cross-entropy
- **Decision boundary**: Linear separation of classes

### Decision Trees
- **Concept**: Recursive binary splits based on features
- **Splitting criteria**: Gini impurity, Information gain
- **Advantages**: Interpretable, handles non-linear relationships

### The Interview Perspective
Interviewers often ask you to:
- Implement gradient descent from scratch
- Explain the math behind these algorithms
- Discuss trade-offs between algorithms

Let's build these algorithms!
    `,problems:["linear-regression-gd","logistic-regression","logistic-regression-full","binary-cross-entropy","decision-tree-split"]},{id:"unsupervised-learning",title:"Unsupervised Learning",description:"Implement clustering and dimensionality reduction algorithms.",icon:"🔍",introduction:`
# Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

## Key Concepts

### K-Means Clustering
- **Goal**: Group data into K clusters
- **Algorithm**:
  1. Initialize K centroids randomly
  2. Assign points to nearest centroid
  3. Update centroids as cluster means
  4. Repeat until convergence

### Principal Component Analysis (PCA)
- **Goal**: Reduce dimensionality while preserving variance
- **Steps**:
  1. Center the data (subtract mean)
  2. Compute covariance matrix
  3. Find eigenvectors/eigenvalues
  4. Project onto top components

### Why These Matter
- **K-Means**: Customer segmentation, image compression
- **PCA**: Feature reduction, visualization, noise removal

### Interview Tips
- Be ready to implement K-means from scratch
- Explain variance explained by components
- Discuss limitations (K-means assumes spherical clusters)

Let's implement these algorithms!
    `,problems:["kmeans-clustering","pca-implementation"]},{id:"model-evaluation",title:"Model Evaluation",description:"Learn metrics and techniques to evaluate ML models properly.",icon:"📈",introduction:`
# Model Evaluation

Building a model is only half the job. You need to evaluate it properly.

## Key Concepts

### Classification Metrics
- **Accuracy**: Correct / Total (can be misleading!)
- **Precision**: TP / (TP + FP) - "Of positive predictions, how many correct?"
- **Recall**: TP / (TP + FN) - "Of actual positives, how many found?"
- **F1 Score**: Harmonic mean of precision and recall

### Regression Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root MSE (same units as target)
- **MAE**: Mean Absolute Error (robust to outliers)
- **R²**: Proportion of variance explained

### Cross-Validation
- **Why**: Single train/test split is unreliable
- **K-Fold**: Split data into K parts, rotate test set
- **Stratified**: Maintain class distribution in folds

### Interview Essentials
- Know when to use each metric
- Implement cross-validation from scratch
- Discuss precision-recall trade-offs

Let's practice evaluation!
    `,problems:["precision-recall-f1","cross-validation","confusion-matrix"]},{id:"deep-learning",title:"Deep Learning Basics",description:"Activation functions and fundamental building blocks.",icon:"⚡",introduction:`
# Deep Learning Basics

Core building blocks that appear in every neural network.

## Key Concepts

### Activation Functions
- **ReLU**: max(0, x) - Most common, avoids vanishing gradient
- **Sigmoid**: 1/(1+e^-x) - Output in [0,1], used for binary classification
- **Softmax**: Converts logits to probabilities that sum to 1
- **Tanh**: Output in [-1, 1], zero-centered

### When to Use Each
- **Hidden layers**: ReLU (or variants like LeakyReLU)
- **Binary output**: Sigmoid
- **Multi-class output**: Softmax
- **RNNs**: Tanh (historically)

### Dense Layers
- **Forward pass**: output = activation(W @ x + b)
- **Parameters**: Weights W and biases b
- **Computation**: Matrix multiplication + bias + activation

Let's implement these fundamentals!
    `,problems:["perceptron","neural-network-forward","backpropagation"]},{id:"neural-networks",title:"Neural Networks",description:"Build neural networks from scratch with forward and backward passes.",icon:"🧠",introduction:`
# Neural Networks

Neural networks are the foundation of deep learning. Understanding the math behind them is essential for ML interviews.

## Key Concepts

### Multi-Layer Perceptron (MLP)
- **Architecture**: Input → Hidden layers → Output
- **Forward pass**: Compute activations layer by layer
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax

### Backpropagation
- **Chain rule**: Compute gradients layer by layer
- **Weight updates**: Gradient descent on all parameters
- **Key insight**: Errors propagate backwards through the network

### Training Techniques
- **Weight Initialization**: Xavier, He initialization
- **Batch Normalization**: Normalize activations for stable training
- **Dropout**: Randomly drop neurons to prevent overfitting

### Loss Functions
- **Cross-entropy**: For classification
- **MSE**: For regression

### Common Interview Topics
- Implement forward/backward pass from scratch
- Explain vanishing/exploding gradients
- Discuss initialization strategies
- Implement regularization techniques

Let's build neural networks from scratch!
    `,problems:["cross-entropy-loss","mlp-forward-backward","weight-init","batch-norm","dropout"]},{id:"cnn",title:"Convolutional Neural Networks",description:"Understand convolutions, pooling, and CNN architectures.",icon:"🖼️",introduction:`
# Convolutional Neural Networks (CNNs)

CNNs are the backbone of computer vision. They learn hierarchical features from images.

## Key Concepts

### Convolution Operation
- **Kernel/Filter**: Small matrix that slides over input
- **Feature maps**: Output of applying filters
- **Parameters**: Stride, padding, kernel size

### Pooling Layers
- **Max pooling**: Take maximum in each region
- **Average pooling**: Take average in each region
- **Purpose**: Reduce spatial dimensions, add translation invariance

### CNN Architecture Components
1. **Convolutional layers**: Extract features
2. **Pooling layers**: Reduce dimensions
3. **Flatten**: Convert 2D to 1D
4. **Fully connected**: Classification

### Output Size Formula
\`\`\`
output = (input - kernel + 2*padding) / stride + 1
\`\`\`

### Classic Architectures
- **LeNet**: First successful CNN
- **AlexNet**: Deep CNNs for ImageNet
- **VGG**: Very deep, small kernels
- **ResNet**: Skip connections

Let's implement CNN operations!
    `,problems:["conv-output-size","conv2d-forward","max-pool","flatten-layer","conv2d-advanced"]},{id:"transformers",title:"Attention & Transformers",description:"Master attention mechanisms and transformer architecture.",icon:"🤖",introduction:`
# Attention and Transformers

Transformers have revolutionized NLP and are now used in vision, speech, and more.

## Key Concepts

### Self-Attention
- **Query, Key, Value**: Three projections of input
- **Attention scores**: Q @ K.T / sqrt(d_k)
- **Output**: Weighted sum of values

### Scaled Dot-Product Attention
\`\`\`
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
\`\`\`

### Multi-Head Attention
- **Multiple heads**: Parallel attention with different projections
- **Concatenate**: Combine all head outputs
- **Project**: Linear transformation back to model dimension

### Transformer Components
- **Positional Encoding**: Inject position information
- **Layer Normalization**: Normalize across features
- **Feed-Forward Network**: MLP after attention
- **Residual Connections**: Add input to output

### Causal Masking
- **Purpose**: Prevent attending to future tokens
- **Implementation**: Add -inf to upper triangle before softmax

### Interview Essentials
- Implement scaled dot-product attention
- Explain why we scale by sqrt(d_k)
- Multi-head attention benefits
- Positional encoding purpose

Let's build transformers from scratch!
    `,problems:["scaled-dot-product-attention","multi-head-attention","positional-encoding","layer-norm","causal-mask"]},{id:"generative-models",title:"Generative Models",description:"Learn VAEs, diffusion models, and generative AI fundamentals.",icon:"🎨",introduction:`
# Generative Models

Generative models learn to create new data similar to the training distribution.

## Key Concepts

### Variational Autoencoders (VAEs)
- **Encoder**: Maps input to latent distribution (μ, σ)
- **Decoder**: Reconstructs input from latent sample
- **Reparameterization trick**: Enable backprop through sampling
- **Loss**: Reconstruction + KL divergence

### VAE Loss (ELBO)
\`\`\`
L = Reconstruction Loss + KL Divergence
L = ||x - x_reconstructed||² + KL(q(z|x) || p(z))
\`\`\`

### Diffusion Models
- **Forward process**: Gradually add noise to data
- **Reverse process**: Learn to denoise
- **Noise schedule**: β_t controls noise at each step

### Key Formulas
\`\`\`
x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
\`\`\`

### KL Divergence
- Measures difference between distributions
- KL(P || Q) ≥ 0, equals 0 iff P = Q
- Not symmetric

### Interview Topics
- Explain reparameterization trick
- VAE loss components
- Diffusion forward/reverse process
- Compare GANs vs VAEs vs Diffusion

Let's implement generative models!
    `,problems:["kl-divergence","vae-reparameterization","vae-loss","vqvae-quantization","diffusion-noise-schedule","diffusion-forward"]},{id:"e2e-implementations",title:"End-to-End Implementations",description:"Build complete ML models from scratch.",icon:"🏗️",introduction:`
# End-to-End Implementations

Put everything together to build complete models from scratch.

## Why Build From Scratch?

1. **Deep Understanding**: Know every component
2. **Interview Preparation**: Common coding questions
3. **Debugging Skills**: Understand what can go wrong
4. **Framework Independence**: Adapt to any library

## Models Covered

### 2-Layer MLP with Backprop
Complete neural network with:
- Forward pass through hidden layer
- Backpropagation of gradients
- Weight updates via gradient descent

### Transformer Encoder
Full encoder block with:
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Feed-forward network
- Residual connections

### Variational Autoencoder (VAE)
Generative model with:
- Encoder → latent distribution
- Reparameterization trick
- Decoder → reconstruction
- ELBO loss (reconstruction + KL)

### Diffusion Model
DDPM-style diffusion with:
- Noise schedule (β, α, ᾱ)
- Forward diffusion process
- Reverse denoising process
- Training objective

### Convolutional Neural Network
Image classifier with:
- Conv2D layers
- Max pooling
- Flatten + FC layers
- Full forward pass

Build these models to truly understand deep learning!
    `,problems:["e2e-mlp","e2e-transformer","e2e-vae","e2e-vqvae","e2e-diffusion","e2e-cnn"]}];function Cv({isOpen:e,onClose:t}){const[n,r]=E.useState(""),[i,a]=E.useState(""),[o,s]=E.useState(!1),[l,u]=E.useState(!1),[d,c]=E.useState("");if(!e)return null;const p=async y=>{y.preventDefault(),s(!0),c("");try{(await fetch("https://formspree.io/f/mpqqwpav",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:n,feedback:i,_subject:`ML Coding Lab Feedback from ${n}`})})).ok?(u(!0),r(""),a("")):c("Failed to send feedback. Please try again.")}catch{c("Failed to send feedback. Please try again.")}finally{s(!1)}},f=()=>{u(!1),c(""),t()};return h.jsxs("div",{className:"fixed inset-0 z-50 flex items-center justify-center",children:[h.jsx("div",{className:"absolute inset-0 bg-black bg-opacity-50",onClick:f}),h.jsxs("div",{className:"relative bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-6",children:[h.jsx("button",{onClick:f,className:"absolute top-4 right-4 text-gray-400 hover:text-gray-600",children:h.jsx("svg",{className:"w-5 h-5",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:h.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M6 18L18 6M6 6l12 12"})})}),h.jsx("h2",{className:"text-xl font-semibold text-gray-900 mb-4",children:"Send Feedback"}),l?h.jsxs("div",{className:"text-center py-8",children:[h.jsx("div",{className:"w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4",children:h.jsx("svg",{className:"w-6 h-6 text-green-600",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:h.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M5 13l4 4L19 7"})})}),h.jsx("p",{className:"text-gray-700 mb-4",children:"Thank you for your feedback!"}),h.jsx("button",{onClick:f,className:"px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors",children:"Close"})]}):h.jsxs("form",{onSubmit:p,className:"space-y-4",children:[h.jsxs("div",{children:[h.jsx("label",{htmlFor:"name",className:"block text-sm font-medium text-gray-700 mb-1",children:"Name"}),h.jsx("input",{type:"text",id:"name",value:n,onChange:y=>r(y.target.value),required:!0,className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent",placeholder:"Your name"})]}),h.jsxs("div",{children:[h.jsx("label",{htmlFor:"feedback",className:"block text-sm font-medium text-gray-700 mb-1",children:"Feedback"}),h.jsx("textarea",{id:"feedback",value:i,onChange:y=>a(y.target.value),required:!0,rows:4,className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none",placeholder:"Share your thoughts, suggestions, or report issues..."})]}),d&&h.jsx("p",{className:"text-red-600 text-sm",children:d}),h.jsxs("div",{className:"flex gap-3 justify-end",children:[h.jsx("button",{type:"button",onClick:f,className:"px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors",children:"Cancel"}),h.jsx("button",{type:"submit",disabled:o,className:"px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:o?"Sending...":"Send Feedback"})]})]})]})]})}const Ev={system:"System UI",spectral:"Spectral",inter:"Inter"},Ls=["system","spectral","inter"];function zv(){const{getOverallProgress:e}=na(),[t,n]=E.useState(!1),[r,i]=E.useState("system");E.useEffect(()=>{document.body.classList.remove("font-system","font-spectral","font-inter"),document.body.classList.add(`font-${r}`)},[r]);const a=()=>{i(l=>{const u=Ls.indexOf(l);return Ls[(u+1)%Ls.length]})},o=Mr.map(l=>({id:l.id,problemCount:l.problems.length})),s=e(o);return h.jsxs(h.Fragment,{children:[h.jsx("header",{className:"bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm",children:h.jsx("div",{className:"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",children:h.jsxs("div",{className:"flex items-center justify-between h-16",children:[h.jsxs(Nt,{to:"/",className:"flex items-center gap-3",children:[h.jsx("img",{src:"/ml-coding-lab/vite.svg",alt:"ML Coding Lab",className:"w-8 h-8"}),h.jsx("span",{className:"text-xl font-semibold text-gray-900",children:"ML Coding Lab"})]}),h.jsxs("div",{className:"flex items-center gap-6",children:[h.jsxs(Nt,{to:"/scratchpad",className:"px-3 py-1.5 bg-gray-100 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-200 transition-colors flex items-center gap-2",children:[h.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:h.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"})}),"Scratchpad"]}),h.jsxs("a",{href:"/ml-coding-lab/ml-cheatsheet.html",target:"_blank",rel:"noopener noreferrer",className:"px-3 py-1.5 bg-amber-100 text-amber-700 text-sm font-medium rounded-md hover:bg-amber-200 transition-colors flex items-center gap-2",children:[h.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:h.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"})}),"Cheat Sheet"]}),h.jsxs("button",{onClick:a,className:"px-3 py-1.5 bg-purple-100 text-purple-700 text-sm font-medium rounded-md hover:bg-purple-200 transition-colors flex items-center gap-2",title:"Toggle font preview",children:[h.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:h.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M4 6h16M4 12h8m-8 6h16"})}),Ev[r]]}),h.jsxs("button",{onClick:()=>n(!0),className:"px-3 py-1.5 text-gray-500 text-sm font-medium hover:text-gray-700 transition-colors flex items-center gap-1",children:[h.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:h.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"})}),"Feedback"]}),h.jsxs("div",{className:"flex items-center gap-3",children:[h.jsx("span",{className:"text-gray-500 text-sm",children:"Overall Progress"}),h.jsx("div",{className:"w-32 h-2 bg-gray-200 rounded-full overflow-hidden",children:h.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${s}%`}})}),h.jsxs("span",{className:"text-primary-600 text-sm font-medium",children:[s,"%"]})]})]})]})})}),h.jsx(Cv,{isOpen:t,onClose:()=>n(!1)})]})}function Tv(){const{getSectionProgress:e}=na();return h.jsx("aside",{className:"w-64 bg-white border-r border-gray-200 min-h-[calc(100vh-4rem)] p-4",children:h.jsxs("nav",{className:"space-y-2",children:[h.jsx("h3",{className:"text-gray-500 text-xs font-semibold uppercase tracking-wider mb-4 px-3",children:"Learning Sections"}),Mr.map((t,n)=>{const r=e(t.id,t.problems.length);return h.jsxs(H1,{to:`/section/${t.id}`,className:({isActive:i})=>`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${i?"bg-primary-50 text-primary-700":"text-gray-700 hover:bg-gray-100 hover:text-gray-900"}`,children:[h.jsx("span",{className:"w-6 h-6 flex items-center justify-center text-lg",children:t.icon}),h.jsxs("div",{className:"flex-1 min-w-0",children:[h.jsxs("div",{className:"flex items-center gap-2",children:[h.jsxs("span",{className:"text-gray-400 text-xs",children:[n+1,"."]}),h.jsx("span",{className:"text-sm font-medium truncate",children:t.title})]}),h.jsxs("div",{className:"flex items-center gap-2 mt-1",children:[h.jsx("div",{className:"flex-1 h-1 bg-gray-200 rounded-full overflow-hidden",children:h.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${r}%`}})}),h.jsxs("span",{className:"text-xs text-gray-400",children:[r,"%"]})]})]})]},t.id)})]})})}function Pv({children:e}){return h.jsxs("div",{className:"min-h-screen bg-gray-50",children:[h.jsx(zv,{}),h.jsxs("div",{className:"flex",children:[h.jsx(Tv,{}),h.jsx("main",{className:"flex-1 p-6",children:e})]})]})}const tp="https://itzsid.github.io/ml-coding-lab",Nv="ML Coding Lab - Learn Machine Learning by Building",Av="Practice hands-on machine learning coding problems. Implement NumPy, neural networks, CNNs, transformers, and more. Run Python code directly in your browser with instant feedback.",Lv="machine learning, ML, deep learning, neural networks, Python, NumPy, transformers, CNN, PyTorch, coding practice, interview prep, AI";function Qo({title:e,description:t=Av,keywords:n=Lv,canonical:r,type:i="website"}){const a=e?`${e} | ML Coding Lab`:Nv,o=r?`${tp}${r}`:tp;return h.jsxs(kv,{children:[h.jsx("title",{children:a}),h.jsx("meta",{name:"title",content:a}),h.jsx("meta",{name:"description",content:t}),h.jsx("meta",{name:"keywords",content:n}),h.jsx("link",{rel:"canonical",href:o}),h.jsx("meta",{property:"og:type",content:i}),h.jsx("meta",{property:"og:url",content:o}),h.jsx("meta",{property:"og:title",content:a}),h.jsx("meta",{property:"og:description",content:t}),h.jsx("meta",{property:"twitter:url",content:o}),h.jsx("meta",{property:"twitter:title",content:a}),h.jsx("meta",{property:"twitter:description",content:t})]})}function jv(){const{getSectionProgress:e,getOverallProgress:t}=na(),n=Mr.map(i=>({id:i.id,problemCount:i.problems.length})),r=t(n);return h.jsxs("div",{className:"max-w-5xl mx-auto",children:[h.jsx(Qo,{canonical:"/"}),h.jsxs("div",{className:"text-center mb-12",children:[h.jsx("h1",{className:"text-4xl font-bold text-gray-900 mb-4",children:"Learn ML By Building"}),h.jsx("p",{className:"text-lg text-gray-600 max-w-2xl mx-auto mb-8",children:"Practice hands-on machine learning coding problems. Run Python code directly in your browser with NumPy and scikit-learn support."}),h.jsxs("div",{className:"inline-flex items-center gap-4 bg-white rounded-lg px-6 py-3 shadow-sm border border-gray-200",children:[h.jsx("span",{className:"text-gray-600",children:"Overall Progress"}),h.jsx("div",{className:"w-48 h-2 bg-gray-200 rounded-full overflow-hidden",children:h.jsx("div",{className:"h-full bg-primary-500 transition-all duration-500",style:{width:`${r}%`}})}),h.jsxs("span",{className:"text-primary-600 font-medium",children:[r,"%"]})]})]}),h.jsx("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:Mr.map((i,a)=>{const o=e(i.id,i.problems.length);return h.jsx(Nt,{to:`/section/${i.id}`,className:"group bg-white rounded-xl p-6 border border-gray-200 hover:border-primary-400 hover:shadow-md transition-all duration-200",children:h.jsxs("div",{className:"flex items-start gap-4",children:[h.jsx("div",{className:"w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center text-2xl group-hover:bg-primary-50 transition-colors",children:i.icon}),h.jsxs("div",{className:"flex-1",children:[h.jsxs("div",{className:"flex items-center gap-2 mb-1",children:[h.jsxs("span",{className:"text-gray-400 text-sm",children:[a+1,"."]}),h.jsx("h3",{className:"text-lg font-semibold text-gray-900 group-hover:text-primary-600 transition-colors",children:i.title})]}),h.jsx("p",{className:"text-gray-500 text-sm mb-4",children:i.description}),h.jsxs("div",{className:"flex items-center justify-between",children:[h.jsxs("span",{className:"text-gray-400 text-sm",children:[i.problems.length," problems"]}),h.jsxs("div",{className:"flex items-center gap-2",children:[h.jsx("div",{className:"w-24 h-1.5 bg-gray-200 rounded-full overflow-hidden",children:h.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${o}%`}})}),h.jsxs("span",{className:"text-xs text-gray-500",children:[o,"%"]})]})]})]})]})},i.id)})}),h.jsxs("div",{className:"mt-16 grid grid-cols-1 md:grid-cols-3 gap-6",children:[h.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[h.jsx("div",{className:"text-3xl mb-3",children:"🐍"}),h.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Python in Browser"}),h.jsx("p",{className:"text-gray-500 text-sm",children:"Run Python code with NumPy, pandas, and more - no setup required."})]}),h.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[h.jsx("div",{className:"text-3xl mb-3",children:"✅"}),h.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Instant Feedback"}),h.jsx("p",{className:"text-gray-500 text-sm",children:"Test your code against multiple test cases and see results immediately."})]}),h.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[h.jsx("div",{className:"text-3xl mb-3",children:"📊"}),h.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Track Progress"}),h.jsx("p",{className:"text-gray-500 text-sm",children:"Your progress is saved locally. Pick up where you left off anytime."})]})]})]})}function Rv(e,t){const n={};return(e[e.length-1]===""?[...e,""]:e).join((n.padRight?" ":"")+","+(n.padLeft===!1?"":" ")).trim()}const Iv=/^[$_\p{ID_Start}][$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,Ov=/^[$_\p{ID_Start}][-$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,Dv={};function np(e,t){return(Dv.jsx?Ov:Iv).test(e)}const Mv=/[ \t\n\f\r]/g;function Fv(e){return typeof e=="object"?e.type==="text"?rp(e.value):!1:rp(e)}function rp(e){return e.replace(Mv,"")===""}class ra{constructor(t,n,r){this.normal=n,this.property=t,r&&(this.space=r)}}ra.prototype.normal={};ra.prototype.property={};ra.prototype.space=void 0;function Bh(e,t){const n={},r={};for(const i of e)Object.assign(n,i.property),Object.assign(r,i.normal);return new ra(n,r,t)}function tu(e){return e.toLowerCase()}class et{constructor(t,n){this.attribute=n,this.property=t}}et.prototype.attribute="";et.prototype.booleanish=!1;et.prototype.boolean=!1;et.prototype.commaOrSpaceSeparated=!1;et.prototype.commaSeparated=!1;et.prototype.defined=!1;et.prototype.mustUseProperty=!1;et.prototype.number=!1;et.prototype.overloadedBoolean=!1;et.prototype.property="";et.prototype.spaceSeparated=!1;et.prototype.space=void 0;let qv=0;const G=tr(),Se=tr(),nu=tr(),R=tr(),de=tr(),Tr=tr(),nt=tr();function tr(){return 2**++qv}const ru=Object.freeze(Object.defineProperty({__proto__:null,boolean:G,booleanish:Se,commaOrSpaceSeparated:nt,commaSeparated:Tr,number:R,overloadedBoolean:nu,spaceSeparated:de},Symbol.toStringTag,{value:"Module"})),js=Object.keys(ru);class dc extends et{constructor(t,n,r,i){let a=-1;if(super(t,n),ip(this,"space",i),typeof r=="number")for(;++a<js.length;){const o=js[a];ip(this,js[a],(r&ru[o])===ru[o])}}}dc.prototype.defined=!0;function ip(e,t,n){n&&(e[t]=n)}function Vr(e){const t={},n={};for(const[r,i]of Object.entries(e.properties)){const a=new dc(r,e.transform(e.attributes||{},r),i,e.space);e.mustUseProperty&&e.mustUseProperty.includes(r)&&(a.mustUseProperty=!0),t[r]=a,n[tu(r)]=r,n[tu(a.attribute)]=r}return new ra(t,n,e.space)}const Uh=Vr({properties:{ariaActiveDescendant:null,ariaAtomic:Se,ariaAutoComplete:null,ariaBusy:Se,ariaChecked:Se,ariaColCount:R,ariaColIndex:R,ariaColSpan:R,ariaControls:de,ariaCurrent:null,ariaDescribedBy:de,ariaDetails:null,ariaDisabled:Se,ariaDropEffect:de,ariaErrorMessage:null,ariaExpanded:Se,ariaFlowTo:de,ariaGrabbed:Se,ariaHasPopup:null,ariaHidden:Se,ariaInvalid:null,ariaKeyShortcuts:null,ariaLabel:null,ariaLabelledBy:de,ariaLevel:R,ariaLive:null,ariaModal:Se,ariaMultiLine:Se,ariaMultiSelectable:Se,ariaOrientation:null,ariaOwns:de,ariaPlaceholder:null,ariaPosInSet:R,ariaPressed:Se,ariaReadOnly:Se,ariaRelevant:null,ariaRequired:Se,ariaRoleDescription:de,ariaRowCount:R,ariaRowIndex:R,ariaRowSpan:R,ariaSelected:Se,ariaSetSize:R,ariaSort:null,ariaValueMax:R,ariaValueMin:R,ariaValueNow:R,ariaValueText:null,role:null},transform(e,t){return t==="role"?t:"aria-"+t.slice(4).toLowerCase()}});function Vh(e,t){return t in e?e[t]:t}function Hh(e,t){return Vh(e,t.toLowerCase())}const Wv=Vr({attributes:{acceptcharset:"accept-charset",classname:"class",htmlfor:"for",httpequiv:"http-equiv"},mustUseProperty:["checked","multiple","muted","selected"],properties:{abbr:null,accept:Tr,acceptCharset:de,accessKey:de,action:null,allow:null,allowFullScreen:G,allowPaymentRequest:G,allowUserMedia:G,alt:null,as:null,async:G,autoCapitalize:null,autoComplete:de,autoFocus:G,autoPlay:G,blocking:de,capture:null,charSet:null,checked:G,cite:null,className:de,cols:R,colSpan:null,content:null,contentEditable:Se,controls:G,controlsList:de,coords:R|Tr,crossOrigin:null,data:null,dateTime:null,decoding:null,default:G,defer:G,dir:null,dirName:null,disabled:G,download:nu,draggable:Se,encType:null,enterKeyHint:null,fetchPriority:null,form:null,formAction:null,formEncType:null,formMethod:null,formNoValidate:G,formTarget:null,headers:de,height:R,hidden:nu,high:R,href:null,hrefLang:null,htmlFor:de,httpEquiv:de,id:null,imageSizes:null,imageSrcSet:null,inert:G,inputMode:null,integrity:null,is:null,isMap:G,itemId:null,itemProp:de,itemRef:de,itemScope:G,itemType:de,kind:null,label:null,lang:null,language:null,list:null,loading:null,loop:G,low:R,manifest:null,max:null,maxLength:R,media:null,method:null,min:null,minLength:R,multiple:G,muted:G,name:null,nonce:null,noModule:G,noValidate:G,onAbort:null,onAfterPrint:null,onAuxClick:null,onBeforeMatch:null,onBeforePrint:null,onBeforeToggle:null,onBeforeUnload:null,onBlur:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onContextLost:null,onContextMenu:null,onContextRestored:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnded:null,onError:null,onFocus:null,onFormData:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLanguageChange:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadEnd:null,onLoadStart:null,onMessage:null,onMessageError:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRejectionHandled:null,onReset:null,onResize:null,onScroll:null,onScrollEnd:null,onSecurityPolicyViolation:null,onSeeked:null,onSeeking:null,onSelect:null,onSlotChange:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnhandledRejection:null,onUnload:null,onVolumeChange:null,onWaiting:null,onWheel:null,open:G,optimum:R,pattern:null,ping:de,placeholder:null,playsInline:G,popover:null,popoverTarget:null,popoverTargetAction:null,poster:null,preload:null,readOnly:G,referrerPolicy:null,rel:de,required:G,reversed:G,rows:R,rowSpan:R,sandbox:de,scope:null,scoped:G,seamless:G,selected:G,shadowRootClonable:G,shadowRootDelegatesFocus:G,shadowRootMode:null,shape:null,size:R,sizes:null,slot:null,span:R,spellCheck:Se,src:null,srcDoc:null,srcLang:null,srcSet:null,start:R,step:null,style:null,tabIndex:R,target:null,title:null,translate:null,type:null,typeMustMatch:G,useMap:null,value:Se,width:R,wrap:null,writingSuggestions:null,align:null,aLink:null,archive:de,axis:null,background:null,bgColor:null,border:R,borderColor:null,bottomMargin:R,cellPadding:null,cellSpacing:null,char:null,charOff:null,classId:null,clear:null,code:null,codeBase:null,codeType:null,color:null,compact:G,declare:G,event:null,face:null,frame:null,frameBorder:null,hSpace:R,leftMargin:R,link:null,longDesc:null,lowSrc:null,marginHeight:R,marginWidth:R,noResize:G,noHref:G,noShade:G,noWrap:G,object:null,profile:null,prompt:null,rev:null,rightMargin:R,rules:null,scheme:null,scrolling:Se,standby:null,summary:null,text:null,topMargin:R,valueType:null,version:null,vAlign:null,vLink:null,vSpace:R,allowTransparency:null,autoCorrect:null,autoSave:null,disablePictureInPicture:G,disableRemotePlayback:G,prefix:null,property:null,results:R,security:null,unselectable:null},space:"html",transform:Hh}),Bv=Vr({attributes:{accentHeight:"accent-height",alignmentBaseline:"alignment-baseline",arabicForm:"arabic-form",baselineShift:"baseline-shift",capHeight:"cap-height",className:"class",clipPath:"clip-path",clipRule:"clip-rule",colorInterpolation:"color-interpolation",colorInterpolationFilters:"color-interpolation-filters",colorProfile:"color-profile",colorRendering:"color-rendering",crossOrigin:"crossorigin",dataType:"datatype",dominantBaseline:"dominant-baseline",enableBackground:"enable-background",fillOpacity:"fill-opacity",fillRule:"fill-rule",floodColor:"flood-color",floodOpacity:"flood-opacity",fontFamily:"font-family",fontSize:"font-size",fontSizeAdjust:"font-size-adjust",fontStretch:"font-stretch",fontStyle:"font-style",fontVariant:"font-variant",fontWeight:"font-weight",glyphName:"glyph-name",glyphOrientationHorizontal:"glyph-orientation-horizontal",glyphOrientationVertical:"glyph-orientation-vertical",hrefLang:"hreflang",horizAdvX:"horiz-adv-x",horizOriginX:"horiz-origin-x",horizOriginY:"horiz-origin-y",imageRendering:"image-rendering",letterSpacing:"letter-spacing",lightingColor:"lighting-color",markerEnd:"marker-end",markerMid:"marker-mid",markerStart:"marker-start",navDown:"nav-down",navDownLeft:"nav-down-left",navDownRight:"nav-down-right",navLeft:"nav-left",navNext:"nav-next",navPrev:"nav-prev",navRight:"nav-right",navUp:"nav-up",navUpLeft:"nav-up-left",navUpRight:"nav-up-right",onAbort:"onabort",onActivate:"onactivate",onAfterPrint:"onafterprint",onBeforePrint:"onbeforeprint",onBegin:"onbegin",onCancel:"oncancel",onCanPlay:"oncanplay",onCanPlayThrough:"oncanplaythrough",onChange:"onchange",onClick:"onclick",onClose:"onclose",onCopy:"oncopy",onCueChange:"oncuechange",onCut:"oncut",onDblClick:"ondblclick",onDrag:"ondrag",onDragEnd:"ondragend",onDragEnter:"ondragenter",onDragExit:"ondragexit",onDragLeave:"ondragleave",onDragOver:"ondragover",onDragStart:"ondragstart",onDrop:"ondrop",onDurationChange:"ondurationchange",onEmptied:"onemptied",onEnd:"onend",onEnded:"onended",onError:"onerror",onFocus:"onfocus",onFocusIn:"onfocusin",onFocusOut:"onfocusout",onHashChange:"onhashchange",onInput:"oninput",onInvalid:"oninvalid",onKeyDown:"onkeydown",onKeyPress:"onkeypress",onKeyUp:"onkeyup",onLoad:"onload",onLoadedData:"onloadeddata",onLoadedMetadata:"onloadedmetadata",onLoadStart:"onloadstart",onMessage:"onmessage",onMouseDown:"onmousedown",onMouseEnter:"onmouseenter",onMouseLeave:"onmouseleave",onMouseMove:"onmousemove",onMouseOut:"onmouseout",onMouseOver:"onmouseover",onMouseUp:"onmouseup",onMouseWheel:"onmousewheel",onOffline:"onoffline",onOnline:"ononline",onPageHide:"onpagehide",onPageShow:"onpageshow",onPaste:"onpaste",onPause:"onpause",onPlay:"onplay",onPlaying:"onplaying",onPopState:"onpopstate",onProgress:"onprogress",onRateChange:"onratechange",onRepeat:"onrepeat",onReset:"onreset",onResize:"onresize",onScroll:"onscroll",onSeeked:"onseeked",onSeeking:"onseeking",onSelect:"onselect",onShow:"onshow",onStalled:"onstalled",onStorage:"onstorage",onSubmit:"onsubmit",onSuspend:"onsuspend",onTimeUpdate:"ontimeupdate",onToggle:"ontoggle",onUnload:"onunload",onVolumeChange:"onvolumechange",onWaiting:"onwaiting",onZoom:"onzoom",overlinePosition:"overline-position",overlineThickness:"overline-thickness",paintOrder:"paint-order",panose1:"panose-1",pointerEvents:"pointer-events",referrerPolicy:"referrerpolicy",renderingIntent:"rendering-intent",shapeRendering:"shape-rendering",stopColor:"stop-color",stopOpacity:"stop-opacity",strikethroughPosition:"strikethrough-position",strikethroughThickness:"strikethrough-thickness",strokeDashArray:"stroke-dasharray",strokeDashOffset:"stroke-dashoffset",strokeLineCap:"stroke-linecap",strokeLineJoin:"stroke-linejoin",strokeMiterLimit:"stroke-miterlimit",strokeOpacity:"stroke-opacity",strokeWidth:"stroke-width",tabIndex:"tabindex",textAnchor:"text-anchor",textDecoration:"text-decoration",textRendering:"text-rendering",transformOrigin:"transform-origin",typeOf:"typeof",underlinePosition:"underline-position",underlineThickness:"underline-thickness",unicodeBidi:"unicode-bidi",unicodeRange:"unicode-range",unitsPerEm:"units-per-em",vAlphabetic:"v-alphabetic",vHanging:"v-hanging",vIdeographic:"v-ideographic",vMathematical:"v-mathematical",vectorEffect:"vector-effect",vertAdvY:"vert-adv-y",vertOriginX:"vert-origin-x",vertOriginY:"vert-origin-y",wordSpacing:"word-spacing",writingMode:"writing-mode",xHeight:"x-height",playbackOrder:"playbackorder",timelineBegin:"timelinebegin"},properties:{about:nt,accentHeight:R,accumulate:null,additive:null,alignmentBaseline:null,alphabetic:R,amplitude:R,arabicForm:null,ascent:R,attributeName:null,attributeType:null,azimuth:R,bandwidth:null,baselineShift:null,baseFrequency:null,baseProfile:null,bbox:null,begin:null,bias:R,by:null,calcMode:null,capHeight:R,className:de,clip:null,clipPath:null,clipPathUnits:null,clipRule:null,color:null,colorInterpolation:null,colorInterpolationFilters:null,colorProfile:null,colorRendering:null,content:null,contentScriptType:null,contentStyleType:null,crossOrigin:null,cursor:null,cx:null,cy:null,d:null,dataType:null,defaultAction:null,descent:R,diffuseConstant:R,direction:null,display:null,dur:null,divisor:R,dominantBaseline:null,download:G,dx:null,dy:null,edgeMode:null,editable:null,elevation:R,enableBackground:null,end:null,event:null,exponent:R,externalResourcesRequired:null,fill:null,fillOpacity:R,fillRule:null,filter:null,filterRes:null,filterUnits:null,floodColor:null,floodOpacity:null,focusable:null,focusHighlight:null,fontFamily:null,fontSize:null,fontSizeAdjust:null,fontStretch:null,fontStyle:null,fontVariant:null,fontWeight:null,format:null,fr:null,from:null,fx:null,fy:null,g1:Tr,g2:Tr,glyphName:Tr,glyphOrientationHorizontal:null,glyphOrientationVertical:null,glyphRef:null,gradientTransform:null,gradientUnits:null,handler:null,hanging:R,hatchContentUnits:null,hatchUnits:null,height:null,href:null,hrefLang:null,horizAdvX:R,horizOriginX:R,horizOriginY:R,id:null,ideographic:R,imageRendering:null,initialVisibility:null,in:null,in2:null,intercept:R,k:R,k1:R,k2:R,k3:R,k4:R,kernelMatrix:nt,kernelUnitLength:null,keyPoints:null,keySplines:null,keyTimes:null,kerning:null,lang:null,lengthAdjust:null,letterSpacing:null,lightingColor:null,limitingConeAngle:R,local:null,markerEnd:null,markerMid:null,markerStart:null,markerHeight:null,markerUnits:null,markerWidth:null,mask:null,maskContentUnits:null,maskUnits:null,mathematical:null,max:null,media:null,mediaCharacterEncoding:null,mediaContentEncodings:null,mediaSize:R,mediaTime:null,method:null,min:null,mode:null,name:null,navDown:null,navDownLeft:null,navDownRight:null,navLeft:null,navNext:null,navPrev:null,navRight:null,navUp:null,navUpLeft:null,navUpRight:null,numOctaves:null,observer:null,offset:null,onAbort:null,onActivate:null,onAfterPrint:null,onBeforePrint:null,onBegin:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnd:null,onEnded:null,onError:null,onFocus:null,onFocusIn:null,onFocusOut:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadStart:null,onMessage:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onMouseWheel:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRepeat:null,onReset:null,onResize:null,onScroll:null,onSeeked:null,onSeeking:null,onSelect:null,onShow:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnload:null,onVolumeChange:null,onWaiting:null,onZoom:null,opacity:null,operator:null,order:null,orient:null,orientation:null,origin:null,overflow:null,overlay:null,overlinePosition:R,overlineThickness:R,paintOrder:null,panose1:null,path:null,pathLength:R,patternContentUnits:null,patternTransform:null,patternUnits:null,phase:null,ping:de,pitch:null,playbackOrder:null,pointerEvents:null,points:null,pointsAtX:R,pointsAtY:R,pointsAtZ:R,preserveAlpha:null,preserveAspectRatio:null,primitiveUnits:null,propagate:null,property:nt,r:null,radius:null,referrerPolicy:null,refX:null,refY:null,rel:nt,rev:nt,renderingIntent:null,repeatCount:null,repeatDur:null,requiredExtensions:nt,requiredFeatures:nt,requiredFonts:nt,requiredFormats:nt,resource:null,restart:null,result:null,rotate:null,rx:null,ry:null,scale:null,seed:null,shapeRendering:null,side:null,slope:null,snapshotTime:null,specularConstant:R,specularExponent:R,spreadMethod:null,spacing:null,startOffset:null,stdDeviation:null,stemh:null,stemv:null,stitchTiles:null,stopColor:null,stopOpacity:null,strikethroughPosition:R,strikethroughThickness:R,string:null,stroke:null,strokeDashArray:nt,strokeDashOffset:null,strokeLineCap:null,strokeLineJoin:null,strokeMiterLimit:R,strokeOpacity:R,strokeWidth:null,style:null,surfaceScale:R,syncBehavior:null,syncBehaviorDefault:null,syncMaster:null,syncTolerance:null,syncToleranceDefault:null,systemLanguage:nt,tabIndex:R,tableValues:null,target:null,targetX:R,targetY:R,textAnchor:null,textDecoration:null,textRendering:null,textLength:null,timelineBegin:null,title:null,transformBehavior:null,type:null,typeOf:nt,to:null,transform:null,transformOrigin:null,u1:null,u2:null,underlinePosition:R,underlineThickness:R,unicode:null,unicodeBidi:null,unicodeRange:null,unitsPerEm:R,values:null,vAlphabetic:R,vMathematical:R,vectorEffect:null,vHanging:R,vIdeographic:R,version:null,vertAdvY:R,vertOriginX:R,vertOriginY:R,viewBox:null,viewTarget:null,visibility:null,width:null,widths:null,wordSpacing:null,writingMode:null,x:null,x1:null,x2:null,xChannelSelector:null,xHeight:R,y:null,y1:null,y2:null,yChannelSelector:null,z:null,zoomAndPan:null},space:"svg",transform:Vh}),$h=Vr({properties:{xLinkActuate:null,xLinkArcRole:null,xLinkHref:null,xLinkRole:null,xLinkShow:null,xLinkTitle:null,xLinkType:null},space:"xlink",transform(e,t){return"xlink:"+t.slice(5).toLowerCase()}}),Qh=Vr({attributes:{xmlnsxlink:"xmlns:xlink"},properties:{xmlnsXLink:null,xmlns:null},space:"xmlns",transform:Hh}),Kh=Vr({properties:{xmlBase:null,xmlLang:null,xmlSpace:null},space:"xml",transform(e,t){return"xml:"+t.slice(3).toLowerCase()}}),Uv={classId:"classID",dataType:"datatype",itemId:"itemID",strokeDashArray:"strokeDasharray",strokeDashOffset:"strokeDashoffset",strokeLineCap:"strokeLinecap",strokeLineJoin:"strokeLinejoin",strokeMiterLimit:"strokeMiterlimit",typeOf:"typeof",xLinkActuate:"xlinkActuate",xLinkArcRole:"xlinkArcrole",xLinkHref:"xlinkHref",xLinkRole:"xlinkRole",xLinkShow:"xlinkShow",xLinkTitle:"xlinkTitle",xLinkType:"xlinkType",xmlnsXLink:"xmlnsXlink"},Vv=/[A-Z]/g,ap=/-[a-z]/g,Hv=/^data[-\w.:]+$/i;function $v(e,t){const n=tu(t);let r=t,i=et;if(n in e.normal)return e.property[e.normal[n]];if(n.length>4&&n.slice(0,4)==="data"&&Hv.test(t)){if(t.charAt(4)==="-"){const a=t.slice(5).replace(ap,Kv);r="data"+a.charAt(0).toUpperCase()+a.slice(1)}else{const a=t.slice(4);if(!ap.test(a)){let o=a.replace(Vv,Qv);o.charAt(0)!=="-"&&(o="-"+o),t="data"+o}}i=dc}return new i(r,t)}function Qv(e){return"-"+e.toLowerCase()}function Kv(e){return e.charAt(1).toUpperCase()}const Xv=Bh([Uh,Wv,$h,Qh,Kh],"html"),pc=Bh([Uh,Bv,$h,Qh,Kh],"svg");function Yv(e){return e.join(" ").trim()}var fc={},op=/\/\*[^*]*\*+([^/*][^*]*\*+)*\//g,Gv=/\n/g,Zv=/^\s*/,Jv=/^(\*?[-#/*\\\w]+(\[[0-9a-z_-]+\])?)\s*/,e_=/^:\s*/,t_=/^((?:'(?:\\'|.)*?'|"(?:\\"|.)*?"|\([^)]*?\)|[^};])+)/,n_=/^[;\s]*/,r_=/^\s+|\s+$/g,i_=`
`,sp="/",lp="*",Fn="",a_="comment",o_="declaration";function s_(e,t){if(typeof e!="string")throw new TypeError("First argument must be a string");if(!e)return[];t=t||{};var n=1,r=1;function i(y){var v=y.match(Gv);v&&(n+=v.length);var k=y.lastIndexOf(i_);r=~k?y.length-k:r+y.length}function a(){var y={line:n,column:r};return function(v){return v.position=new o(y),u(),v}}function o(y){this.start=y,this.end={line:n,column:r},this.source=t.source}o.prototype.content=e;function s(y){var v=new Error(t.source+":"+n+":"+r+": "+y);if(v.reason=y,v.filename=t.source,v.line=n,v.column=r,v.source=e,!t.silent)throw v}function l(y){var v=y.exec(e);if(v){var k=v[0];return i(k),e=e.slice(k.length),v}}function u(){l(Zv)}function d(y){var v;for(y=y||[];v=c();)v!==!1&&y.push(v);return y}function c(){var y=a();if(!(sp!=e.charAt(0)||lp!=e.charAt(1))){for(var v=2;Fn!=e.charAt(v)&&(lp!=e.charAt(v)||sp!=e.charAt(v+1));)++v;if(v+=2,Fn===e.charAt(v-1))return s("End of comment missing");var k=e.slice(2,v-2);return r+=2,i(k),e=e.slice(v),r+=2,y({type:a_,comment:k})}}function p(){var y=a(),v=l(Jv);if(v){if(c(),!l(e_))return s("property missing ':'");var k=l(t_),m=y({type:o_,property:up(v[0].replace(op,Fn)),value:k?up(k[0].replace(op,Fn)):Fn});return l(n_),m}}function f(){var y=[];d(y);for(var v;v=p();)v!==!1&&(y.push(v),d(y));return y}return u(),f()}function up(e){return e?e.replace(r_,Fn):Fn}var l_=s_,u_=Xa&&Xa.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(fc,"__esModule",{value:!0});fc.default=d_;const c_=u_(l_);function d_(e,t){let n=null;if(!e||typeof e!="string")return n;const r=(0,c_.default)(e),i=typeof t=="function";return r.forEach(a=>{if(a.type!=="declaration")return;const{property:o,value:s}=a;i?t(o,s,a):s&&(n=n||{},n[o]=s)}),n}var Ko={};Object.defineProperty(Ko,"__esModule",{value:!0});Ko.camelCase=void 0;var p_=/^--[a-zA-Z0-9_-]+$/,f_=/-([a-z])/g,m_=/^[^-]+$/,h_=/^-(webkit|moz|ms|o|khtml)-/,g_=/^-(ms)-/,y_=function(e){return!e||m_.test(e)||p_.test(e)},x_=function(e,t){return t.toUpperCase()},cp=function(e,t){return"".concat(t,"-")},v_=function(e,t){return t===void 0&&(t={}),y_(e)?e:(e=e.toLowerCase(),t.reactCompat?e=e.replace(g_,cp):e=e.replace(h_,cp),e.replace(f_,x_))};Ko.camelCase=v_;var __=Xa&&Xa.__importDefault||function(e){return e&&e.__esModule?e:{default:e}},w_=__(fc),b_=Ko;function iu(e,t){var n={};return!e||typeof e!="string"||(0,w_.default)(e,function(r,i){r&&i&&(n[(0,b_.camelCase)(r,t)]=i)}),n}iu.default=iu;var k_=iu;const S_=Zn(k_),Xh=Yh("end"),mc=Yh("start");function Yh(e){return t;function t(n){const r=n&&n.position&&n.position[e]||{};if(typeof r.line=="number"&&r.line>0&&typeof r.column=="number"&&r.column>0)return{line:r.line,column:r.column,offset:typeof r.offset=="number"&&r.offset>-1?r.offset:void 0}}}function C_(e){const t=mc(e),n=Xh(e);if(t&&n)return{start:t,end:n}}function bi(e){return!e||typeof e!="object"?"":"position"in e||"type"in e?dp(e.position):"start"in e||"end"in e?dp(e):"line"in e||"column"in e?au(e):""}function au(e){return pp(e&&e.line)+":"+pp(e&&e.column)}function dp(e){return au(e&&e.start)+"-"+au(e&&e.end)}function pp(e){return e&&typeof e=="number"?e:1}class We extends Error{constructor(t,n,r){super(),typeof n=="string"&&(r=n,n=void 0);let i="",a={},o=!1;if(n&&("line"in n&&"column"in n?a={place:n}:"start"in n&&"end"in n?a={place:n}:"type"in n?a={ancestors:[n],place:n.position}:a={...n}),typeof t=="string"?i=t:!a.cause&&t&&(o=!0,i=t.message,a.cause=t),!a.ruleId&&!a.source&&typeof r=="string"){const l=r.indexOf(":");l===-1?a.ruleId=r:(a.source=r.slice(0,l),a.ruleId=r.slice(l+1))}if(!a.place&&a.ancestors&&a.ancestors){const l=a.ancestors[a.ancestors.length-1];l&&(a.place=l.position)}const s=a.place&&"start"in a.place?a.place.start:a.place;this.ancestors=a.ancestors||void 0,this.cause=a.cause||void 0,this.column=s?s.column:void 0,this.fatal=void 0,this.file="",this.message=i,this.line=s?s.line:void 0,this.name=bi(a.place)||"1:1",this.place=a.place||void 0,this.reason=this.message,this.ruleId=a.ruleId||void 0,this.source=a.source||void 0,this.stack=o&&a.cause&&typeof a.cause.stack=="string"?a.cause.stack:"",this.actual=void 0,this.expected=void 0,this.note=void 0,this.url=void 0}}We.prototype.file="";We.prototype.name="";We.prototype.reason="";We.prototype.message="";We.prototype.stack="";We.prototype.column=void 0;We.prototype.line=void 0;We.prototype.ancestors=void 0;We.prototype.cause=void 0;We.prototype.fatal=void 0;We.prototype.place=void 0;We.prototype.ruleId=void 0;We.prototype.source=void 0;const hc={}.hasOwnProperty,E_=new Map,z_=/[A-Z]/g,T_=new Set(["table","tbody","thead","tfoot","tr"]),P_=new Set(["td","th"]),Gh="https://github.com/syntax-tree/hast-util-to-jsx-runtime";function N_(e,t){if(!t||t.Fragment===void 0)throw new TypeError("Expected `Fragment` in options");const n=t.filePath||void 0;let r;if(t.development){if(typeof t.jsxDEV!="function")throw new TypeError("Expected `jsxDEV` in options when `development: true`");r=M_(n,t.jsxDEV)}else{if(typeof t.jsx!="function")throw new TypeError("Expected `jsx` in production options");if(typeof t.jsxs!="function")throw new TypeError("Expected `jsxs` in production options");r=D_(n,t.jsx,t.jsxs)}const i={Fragment:t.Fragment,ancestors:[],components:t.components||{},create:r,elementAttributeNameCase:t.elementAttributeNameCase||"react",evaluater:t.createEvaluater?t.createEvaluater():void 0,filePath:n,ignoreInvalidStyle:t.ignoreInvalidStyle||!1,passKeys:t.passKeys!==!1,passNode:t.passNode||!1,schema:t.space==="svg"?pc:Xv,stylePropertyNameCase:t.stylePropertyNameCase||"dom",tableCellAlignToStyle:t.tableCellAlignToStyle!==!1},a=Zh(i,e,void 0);return a&&typeof a!="string"?a:i.create(e,i.Fragment,{children:a||void 0},void 0)}function Zh(e,t,n){if(t.type==="element")return A_(e,t,n);if(t.type==="mdxFlowExpression"||t.type==="mdxTextExpression")return L_(e,t);if(t.type==="mdxJsxFlowElement"||t.type==="mdxJsxTextElement")return R_(e,t,n);if(t.type==="mdxjsEsm")return j_(e,t);if(t.type==="root")return I_(e,t,n);if(t.type==="text")return O_(e,t)}function A_(e,t,n){const r=e.schema;let i=r;t.tagName.toLowerCase()==="svg"&&r.space==="html"&&(i=pc,e.schema=i),e.ancestors.push(t);const a=eg(e,t.tagName,!1),o=F_(e,t);let s=yc(e,t);return T_.has(t.tagName)&&(s=s.filter(function(l){return typeof l=="string"?!Fv(l):!0})),Jh(e,o,a,t),gc(o,s),e.ancestors.pop(),e.schema=r,e.create(t,a,o,n)}function L_(e,t){if(t.data&&t.data.estree&&e.evaluater){const r=t.data.estree.body[0];return r.type,e.evaluater.evaluateExpression(r.expression)}Hi(e,t.position)}function j_(e,t){if(t.data&&t.data.estree&&e.evaluater)return e.evaluater.evaluateProgram(t.data.estree);Hi(e,t.position)}function R_(e,t,n){const r=e.schema;let i=r;t.name==="svg"&&r.space==="html"&&(i=pc,e.schema=i),e.ancestors.push(t);const a=t.name===null?e.Fragment:eg(e,t.name,!0),o=q_(e,t),s=yc(e,t);return Jh(e,o,a,t),gc(o,s),e.ancestors.pop(),e.schema=r,e.create(t,a,o,n)}function I_(e,t,n){const r={};return gc(r,yc(e,t)),e.create(t,e.Fragment,r,n)}function O_(e,t){return t.value}function Jh(e,t,n,r){typeof n!="string"&&n!==e.Fragment&&e.passNode&&(t.node=r)}function gc(e,t){if(t.length>0){const n=t.length>1?t:t[0];n&&(e.children=n)}}function D_(e,t,n){return r;function r(i,a,o,s){const u=Array.isArray(o.children)?n:t;return s?u(a,o,s):u(a,o)}}function M_(e,t){return n;function n(r,i,a,o){const s=Array.isArray(a.children),l=mc(r);return t(i,a,o,s,{columnNumber:l?l.column-1:void 0,fileName:e,lineNumber:l?l.line:void 0},void 0)}}function F_(e,t){const n={};let r,i;for(i in t.properties)if(i!=="children"&&hc.call(t.properties,i)){const a=W_(e,i,t.properties[i]);if(a){const[o,s]=a;e.tableCellAlignToStyle&&o==="align"&&typeof s=="string"&&P_.has(t.tagName)?r=s:n[o]=s}}if(r){const a=n.style||(n.style={});a[e.stylePropertyNameCase==="css"?"text-align":"textAlign"]=r}return n}function q_(e,t){const n={};for(const r of t.attributes)if(r.type==="mdxJsxExpressionAttribute")if(r.data&&r.data.estree&&e.evaluater){const a=r.data.estree.body[0];a.type;const o=a.expression;o.type;const s=o.properties[0];s.type,Object.assign(n,e.evaluater.evaluateExpression(s.argument))}else Hi(e,t.position);else{const i=r.name;let a;if(r.value&&typeof r.value=="object")if(r.value.data&&r.value.data.estree&&e.evaluater){const s=r.value.data.estree.body[0];s.type,a=e.evaluater.evaluateExpression(s.expression)}else Hi(e,t.position);else a=r.value===null?!0:r.value;n[i]=a}return n}function yc(e,t){const n=[];let r=-1;const i=e.passKeys?new Map:E_;for(;++r<t.children.length;){const a=t.children[r];let o;if(e.passKeys){const l=a.type==="element"?a.tagName:a.type==="mdxJsxFlowElement"||a.type==="mdxJsxTextElement"?a.name:void 0;if(l){const u=i.get(l)||0;o=l+"-"+u,i.set(l,u+1)}}const s=Zh(e,a,o);s!==void 0&&n.push(s)}return n}function W_(e,t,n){const r=$v(e.schema,t);if(!(n==null||typeof n=="number"&&Number.isNaN(n))){if(Array.isArray(n)&&(n=r.commaSeparated?Rv(n):Yv(n)),r.property==="style"){let i=typeof n=="object"?n:B_(e,String(n));return e.stylePropertyNameCase==="css"&&(i=U_(i)),["style",i]}return[e.elementAttributeNameCase==="react"&&r.space?Uv[r.property]||r.property:r.attribute,n]}}function B_(e,t){try{return S_(t,{reactCompat:!0})}catch(n){if(e.ignoreInvalidStyle)return{};const r=n,i=new We("Cannot parse `style` attribute",{ancestors:e.ancestors,cause:r,ruleId:"style",source:"hast-util-to-jsx-runtime"});throw i.file=e.filePath||void 0,i.url=Gh+"#cannot-parse-style-attribute",i}}function eg(e,t,n){let r;if(!n)r={type:"Literal",value:t};else if(t.includes(".")){const i=t.split(".");let a=-1,o;for(;++a<i.length;){const s=np(i[a])?{type:"Identifier",name:i[a]}:{type:"Literal",value:i[a]};o=o?{type:"MemberExpression",object:o,property:s,computed:!!(a&&s.type==="Literal"),optional:!1}:s}r=o}else r=np(t)&&!/^[a-z]/.test(t)?{type:"Identifier",name:t}:{type:"Literal",value:t};if(r.type==="Literal"){const i=r.value;return hc.call(e.components,i)?e.components[i]:i}if(e.evaluater)return e.evaluater.evaluateExpression(r);Hi(e)}function Hi(e,t){const n=new We("Cannot handle MDX estrees without `createEvaluater`",{ancestors:e.ancestors,place:t,ruleId:"mdx-estree",source:"hast-util-to-jsx-runtime"});throw n.file=e.filePath||void 0,n.url=Gh+"#cannot-handle-mdx-estrees-without-createevaluater",n}function U_(e){const t={};let n;for(n in e)hc.call(e,n)&&(t[V_(n)]=e[n]);return t}function V_(e){let t=e.replace(z_,H_);return t.slice(0,3)==="ms-"&&(t="-"+t),t}function H_(e){return"-"+e.toLowerCase()}const Rs={action:["form"],cite:["blockquote","del","ins","q"],data:["object"],formAction:["button","input"],href:["a","area","base","link"],icon:["menuitem"],itemId:null,manifest:["html"],ping:["a","area"],poster:["video"],src:["audio","embed","iframe","img","input","script","source","track","video"]},$_={};function xc(e,t){const n=$_,r=typeof n.includeImageAlt=="boolean"?n.includeImageAlt:!0,i=typeof n.includeHtml=="boolean"?n.includeHtml:!0;return tg(e,r,i)}function tg(e,t,n){if(Q_(e)){if("value"in e)return e.type==="html"&&!n?"":e.value;if(t&&"alt"in e&&e.alt)return e.alt;if("children"in e)return fp(e.children,t,n)}return Array.isArray(e)?fp(e,t,n):""}function fp(e,t,n){const r=[];let i=-1;for(;++i<e.length;)r[i]=tg(e[i],t,n);return r.join("")}function Q_(e){return!!(e&&typeof e=="object")}const mp=document.createElement("i");function vc(e){const t="&"+e+";";mp.innerHTML=t;const n=mp.textContent;return n.charCodeAt(n.length-1)===59&&e!=="semi"||n===t?!1:n}function st(e,t,n,r){const i=e.length;let a=0,o;if(t<0?t=-t>i?0:i+t:t=t>i?i:t,n=n>0?n:0,r.length<1e4)o=Array.from(r),o.unshift(t,n),e.splice(...o);else for(n&&e.splice(t,n);a<r.length;)o=r.slice(a,a+1e4),o.unshift(t,0),e.splice(...o),a+=1e4,t+=1e4}function gt(e,t){return e.length>0?(st(e,e.length,0,t),e):t}const hp={}.hasOwnProperty;function ng(e){const t={};let n=-1;for(;++n<e.length;)K_(t,e[n]);return t}function K_(e,t){let n;for(n in t){const i=(hp.call(e,n)?e[n]:void 0)||(e[n]={}),a=t[n];let o;if(a)for(o in a){hp.call(i,o)||(i[o]=[]);const s=a[o];X_(i[o],Array.isArray(s)?s:s?[s]:[])}}}function X_(e,t){let n=-1;const r=[];for(;++n<t.length;)(t[n].add==="after"?e:r).push(t[n]);st(e,0,0,r)}function rg(e,t){const n=Number.parseInt(e,t);return n<9||n===11||n>13&&n<32||n>126&&n<160||n>55295&&n<57344||n>64975&&n<65008||(n&65535)===65535||(n&65535)===65534||n>1114111?"�":String.fromCodePoint(n)}function Rt(e){return e.replace(/[\t\n\r ]+/g," ").replace(/^ | $/g,"").toLowerCase().toUpperCase()}const Ve=Ln(/[A-Za-z]/),Fe=Ln(/[\dA-Za-z]/),Y_=Ln(/[#-'*+\--9=?A-Z^-~]/);function ko(e){return e!==null&&(e<32||e===127)}const ou=Ln(/\d/),G_=Ln(/[\dA-Fa-f]/),Z_=Ln(/[!-/:-@[-`{-~]/);function K(e){return e!==null&&e<-2}function ce(e){return e!==null&&(e<0||e===32)}function Z(e){return e===-2||e===-1||e===32}const Xo=Ln(new RegExp("\\p{P}|\\p{S}","u")),Gn=Ln(/\s/);function Ln(e){return t;function t(n){return n!==null&&n>-1&&e.test(String.fromCharCode(n))}}function Hr(e){const t=[];let n=-1,r=0,i=0;for(;++n<e.length;){const a=e.charCodeAt(n);let o="";if(a===37&&Fe(e.charCodeAt(n+1))&&Fe(e.charCodeAt(n+2)))i=2;else if(a<128)/[!#$&-;=?-Z_a-z~]/.test(String.fromCharCode(a))||(o=String.fromCharCode(a));else if(a>55295&&a<57344){const s=e.charCodeAt(n+1);a<56320&&s>56319&&s<57344?(o=String.fromCharCode(a,s),i=1):o="�"}else o=String.fromCharCode(a);o&&(t.push(e.slice(r,n),encodeURIComponent(o)),r=n+i+1,o=""),i&&(n+=i,i=0)}return t.join("")+e.slice(r)}function re(e,t,n,r){const i=r?r-1:Number.POSITIVE_INFINITY;let a=0;return o;function o(l){return Z(l)?(e.enter(n),s(l)):t(l)}function s(l){return Z(l)&&a++<i?(e.consume(l),s):(e.exit(n),t(l))}}const J_={tokenize:ew};function ew(e){const t=e.attempt(this.parser.constructs.contentInitial,r,i);let n;return t;function r(s){if(s===null){e.consume(s);return}return e.enter("lineEnding"),e.consume(s),e.exit("lineEnding"),re(e,t,"linePrefix")}function i(s){return e.enter("paragraph"),a(s)}function a(s){const l=e.enter("chunkText",{contentType:"text",previous:n});return n&&(n.next=l),n=l,o(s)}function o(s){if(s===null){e.exit("chunkText"),e.exit("paragraph"),e.consume(s);return}return K(s)?(e.consume(s),e.exit("chunkText"),a):(e.consume(s),o)}}const tw={tokenize:nw},gp={tokenize:rw};function nw(e){const t=this,n=[];let r=0,i,a,o;return s;function s(x){if(r<n.length){const C=n[r];return t.containerState=C[1],e.attempt(C[0].continuation,l,u)(x)}return u(x)}function l(x){if(r++,t.containerState._closeFlow){t.containerState._closeFlow=void 0,i&&g();const C=t.events.length;let z=C,S;for(;z--;)if(t.events[z][0]==="exit"&&t.events[z][1].type==="chunkFlow"){S=t.events[z][1].end;break}m(r);let T=C;for(;T<t.events.length;)t.events[T][1].end={...S},T++;return st(t.events,z+1,0,t.events.slice(C)),t.events.length=T,u(x)}return s(x)}function u(x){if(r===n.length){if(!i)return p(x);if(i.currentConstruct&&i.currentConstruct.concrete)return y(x);t.interrupt=!!(i.currentConstruct&&!i._gfmTableDynamicInterruptHack)}return t.containerState={},e.check(gp,d,c)(x)}function d(x){return i&&g(),m(r),p(x)}function c(x){return t.parser.lazy[t.now().line]=r!==n.length,o=t.now().offset,y(x)}function p(x){return t.containerState={},e.attempt(gp,f,y)(x)}function f(x){return r++,n.push([t.currentConstruct,t.containerState]),p(x)}function y(x){if(x===null){i&&g(),m(0),e.consume(x);return}return i=i||t.parser.flow(t.now()),e.enter("chunkFlow",{_tokenizer:i,contentType:"flow",previous:a}),v(x)}function v(x){if(x===null){k(e.exit("chunkFlow"),!0),m(0),e.consume(x);return}return K(x)?(e.consume(x),k(e.exit("chunkFlow")),r=0,t.interrupt=void 0,s):(e.consume(x),v)}function k(x,C){const z=t.sliceStream(x);if(C&&z.push(null),x.previous=a,a&&(a.next=x),a=x,i.defineSkip(x.start),i.write(z),t.parser.lazy[x.start.line]){let S=i.events.length;for(;S--;)if(i.events[S][1].start.offset<o&&(!i.events[S][1].end||i.events[S][1].end.offset>o))return;const T=t.events.length;let N=T,F,b;for(;N--;)if(t.events[N][0]==="exit"&&t.events[N][1].type==="chunkFlow"){if(F){b=t.events[N][1].end;break}F=!0}for(m(r),S=T;S<t.events.length;)t.events[S][1].end={...b},S++;st(t.events,N+1,0,t.events.slice(T)),t.events.length=S}}function m(x){let C=n.length;for(;C-- >x;){const z=n[C];t.containerState=z[1],z[0].exit.call(t,e)}n.length=x}function g(){i.write([null]),a=void 0,i=void 0,t.containerState._closeFlow=void 0}}function rw(e,t,n){return re(e,e.attempt(this.parser.constructs.document,t,n),"linePrefix",this.parser.constructs.disable.null.includes("codeIndented")?void 0:4)}function Fr(e){if(e===null||ce(e)||Gn(e))return 1;if(Xo(e))return 2}function Yo(e,t,n){const r=[];let i=-1;for(;++i<e.length;){const a=e[i].resolveAll;a&&!r.includes(a)&&(t=a(t,n),r.push(a))}return t}const su={name:"attention",resolveAll:iw,tokenize:aw};function iw(e,t){let n=-1,r,i,a,o,s,l,u,d;for(;++n<e.length;)if(e[n][0]==="enter"&&e[n][1].type==="attentionSequence"&&e[n][1]._close){for(r=n;r--;)if(e[r][0]==="exit"&&e[r][1].type==="attentionSequence"&&e[r][1]._open&&t.sliceSerialize(e[r][1]).charCodeAt(0)===t.sliceSerialize(e[n][1]).charCodeAt(0)){if((e[r][1]._close||e[n][1]._open)&&(e[n][1].end.offset-e[n][1].start.offset)%3&&!((e[r][1].end.offset-e[r][1].start.offset+e[n][1].end.offset-e[n][1].start.offset)%3))continue;l=e[r][1].end.offset-e[r][1].start.offset>1&&e[n][1].end.offset-e[n][1].start.offset>1?2:1;const c={...e[r][1].end},p={...e[n][1].start};yp(c,-l),yp(p,l),o={type:l>1?"strongSequence":"emphasisSequence",start:c,end:{...e[r][1].end}},s={type:l>1?"strongSequence":"emphasisSequence",start:{...e[n][1].start},end:p},a={type:l>1?"strongText":"emphasisText",start:{...e[r][1].end},end:{...e[n][1].start}},i={type:l>1?"strong":"emphasis",start:{...o.start},end:{...s.end}},e[r][1].end={...o.start},e[n][1].start={...s.end},u=[],e[r][1].end.offset-e[r][1].start.offset&&(u=gt(u,[["enter",e[r][1],t],["exit",e[r][1],t]])),u=gt(u,[["enter",i,t],["enter",o,t],["exit",o,t],["enter",a,t]]),u=gt(u,Yo(t.parser.constructs.insideSpan.null,e.slice(r+1,n),t)),u=gt(u,[["exit",a,t],["enter",s,t],["exit",s,t],["exit",i,t]]),e[n][1].end.offset-e[n][1].start.offset?(d=2,u=gt(u,[["enter",e[n][1],t],["exit",e[n][1],t]])):d=0,st(e,r-1,n-r+3,u),n=r+u.length-d-2;break}}for(n=-1;++n<e.length;)e[n][1].type==="attentionSequence"&&(e[n][1].type="data");return e}function aw(e,t){const n=this.parser.constructs.attentionMarkers.null,r=this.previous,i=Fr(r);let a;return o;function o(l){return a=l,e.enter("attentionSequence"),s(l)}function s(l){if(l===a)return e.consume(l),s;const u=e.exit("attentionSequence"),d=Fr(l),c=!d||d===2&&i||n.includes(l),p=!i||i===2&&d||n.includes(r);return u._open=!!(a===42?c:c&&(i||!p)),u._close=!!(a===42?p:p&&(d||!c)),t(l)}}function yp(e,t){e.column+=t,e.offset+=t,e._bufferIndex+=t}const ow={name:"autolink",tokenize:sw};function sw(e,t,n){let r=0;return i;function i(f){return e.enter("autolink"),e.enter("autolinkMarker"),e.consume(f),e.exit("autolinkMarker"),e.enter("autolinkProtocol"),a}function a(f){return Ve(f)?(e.consume(f),o):f===64?n(f):u(f)}function o(f){return f===43||f===45||f===46||Fe(f)?(r=1,s(f)):u(f)}function s(f){return f===58?(e.consume(f),r=0,l):(f===43||f===45||f===46||Fe(f))&&r++<32?(e.consume(f),s):(r=0,u(f))}function l(f){return f===62?(e.exit("autolinkProtocol"),e.enter("autolinkMarker"),e.consume(f),e.exit("autolinkMarker"),e.exit("autolink"),t):f===null||f===32||f===60||ko(f)?n(f):(e.consume(f),l)}function u(f){return f===64?(e.consume(f),d):Y_(f)?(e.consume(f),u):n(f)}function d(f){return Fe(f)?c(f):n(f)}function c(f){return f===46?(e.consume(f),r=0,d):f===62?(e.exit("autolinkProtocol").type="autolinkEmail",e.enter("autolinkMarker"),e.consume(f),e.exit("autolinkMarker"),e.exit("autolink"),t):p(f)}function p(f){if((f===45||Fe(f))&&r++<63){const y=f===45?p:c;return e.consume(f),y}return n(f)}}const ia={partial:!0,tokenize:lw};function lw(e,t,n){return r;function r(a){return Z(a)?re(e,i,"linePrefix")(a):i(a)}function i(a){return a===null||K(a)?t(a):n(a)}}const ig={continuation:{tokenize:cw},exit:dw,name:"blockQuote",tokenize:uw};function uw(e,t,n){const r=this;return i;function i(o){if(o===62){const s=r.containerState;return s.open||(e.enter("blockQuote",{_container:!0}),s.open=!0),e.enter("blockQuotePrefix"),e.enter("blockQuoteMarker"),e.consume(o),e.exit("blockQuoteMarker"),a}return n(o)}function a(o){return Z(o)?(e.enter("blockQuotePrefixWhitespace"),e.consume(o),e.exit("blockQuotePrefixWhitespace"),e.exit("blockQuotePrefix"),t):(e.exit("blockQuotePrefix"),t(o))}}function cw(e,t,n){const r=this;return i;function i(o){return Z(o)?re(e,a,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(o):a(o)}function a(o){return e.attempt(ig,t,n)(o)}}function dw(e){e.exit("blockQuote")}const ag={name:"characterEscape",tokenize:pw};function pw(e,t,n){return r;function r(a){return e.enter("characterEscape"),e.enter("escapeMarker"),e.consume(a),e.exit("escapeMarker"),i}function i(a){return Z_(a)?(e.enter("characterEscapeValue"),e.consume(a),e.exit("characterEscapeValue"),e.exit("characterEscape"),t):n(a)}}const og={name:"characterReference",tokenize:fw};function fw(e,t,n){const r=this;let i=0,a,o;return s;function s(c){return e.enter("characterReference"),e.enter("characterReferenceMarker"),e.consume(c),e.exit("characterReferenceMarker"),l}function l(c){return c===35?(e.enter("characterReferenceMarkerNumeric"),e.consume(c),e.exit("characterReferenceMarkerNumeric"),u):(e.enter("characterReferenceValue"),a=31,o=Fe,d(c))}function u(c){return c===88||c===120?(e.enter("characterReferenceMarkerHexadecimal"),e.consume(c),e.exit("characterReferenceMarkerHexadecimal"),e.enter("characterReferenceValue"),a=6,o=G_,d):(e.enter("characterReferenceValue"),a=7,o=ou,d(c))}function d(c){if(c===59&&i){const p=e.exit("characterReferenceValue");return o===Fe&&!vc(r.sliceSerialize(p))?n(c):(e.enter("characterReferenceMarker"),e.consume(c),e.exit("characterReferenceMarker"),e.exit("characterReference"),t)}return o(c)&&i++<a?(e.consume(c),d):n(c)}}const xp={partial:!0,tokenize:hw},vp={concrete:!0,name:"codeFenced",tokenize:mw};function mw(e,t,n){const r=this,i={partial:!0,tokenize:z};let a=0,o=0,s;return l;function l(S){return u(S)}function u(S){const T=r.events[r.events.length-1];return a=T&&T[1].type==="linePrefix"?T[2].sliceSerialize(T[1],!0).length:0,s=S,e.enter("codeFenced"),e.enter("codeFencedFence"),e.enter("codeFencedFenceSequence"),d(S)}function d(S){return S===s?(o++,e.consume(S),d):o<3?n(S):(e.exit("codeFencedFenceSequence"),Z(S)?re(e,c,"whitespace")(S):c(S))}function c(S){return S===null||K(S)?(e.exit("codeFencedFence"),r.interrupt?t(S):e.check(xp,v,C)(S)):(e.enter("codeFencedFenceInfo"),e.enter("chunkString",{contentType:"string"}),p(S))}function p(S){return S===null||K(S)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),c(S)):Z(S)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),re(e,f,"whitespace")(S)):S===96&&S===s?n(S):(e.consume(S),p)}function f(S){return S===null||K(S)?c(S):(e.enter("codeFencedFenceMeta"),e.enter("chunkString",{contentType:"string"}),y(S))}function y(S){return S===null||K(S)?(e.exit("chunkString"),e.exit("codeFencedFenceMeta"),c(S)):S===96&&S===s?n(S):(e.consume(S),y)}function v(S){return e.attempt(i,C,k)(S)}function k(S){return e.enter("lineEnding"),e.consume(S),e.exit("lineEnding"),m}function m(S){return a>0&&Z(S)?re(e,g,"linePrefix",a+1)(S):g(S)}function g(S){return S===null||K(S)?e.check(xp,v,C)(S):(e.enter("codeFlowValue"),x(S))}function x(S){return S===null||K(S)?(e.exit("codeFlowValue"),g(S)):(e.consume(S),x)}function C(S){return e.exit("codeFenced"),t(S)}function z(S,T,N){let F=0;return b;function b(A){return S.enter("lineEnding"),S.consume(A),S.exit("lineEnding"),I}function I(A){return S.enter("codeFencedFence"),Z(A)?re(S,W,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(A):W(A)}function W(A){return A===s?(S.enter("codeFencedFenceSequence"),X(A)):N(A)}function X(A){return A===s?(F++,S.consume(A),X):F>=o?(S.exit("codeFencedFenceSequence"),Z(A)?re(S,ee,"whitespace")(A):ee(A)):N(A)}function ee(A){return A===null||K(A)?(S.exit("codeFencedFence"),T(A)):N(A)}}}function hw(e,t,n){const r=this;return i;function i(o){return o===null?n(o):(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),a)}function a(o){return r.parser.lazy[r.now().line]?n(o):t(o)}}const Is={name:"codeIndented",tokenize:yw},gw={partial:!0,tokenize:xw};function yw(e,t,n){const r=this;return i;function i(u){return e.enter("codeIndented"),re(e,a,"linePrefix",5)(u)}function a(u){const d=r.events[r.events.length-1];return d&&d[1].type==="linePrefix"&&d[2].sliceSerialize(d[1],!0).length>=4?o(u):n(u)}function o(u){return u===null?l(u):K(u)?e.attempt(gw,o,l)(u):(e.enter("codeFlowValue"),s(u))}function s(u){return u===null||K(u)?(e.exit("codeFlowValue"),o(u)):(e.consume(u),s)}function l(u){return e.exit("codeIndented"),t(u)}}function xw(e,t,n){const r=this;return i;function i(o){return r.parser.lazy[r.now().line]?n(o):K(o)?(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),i):re(e,a,"linePrefix",5)(o)}function a(o){const s=r.events[r.events.length-1];return s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(o):K(o)?i(o):n(o)}}const vw={name:"codeText",previous:ww,resolve:_w,tokenize:bw};function _w(e){let t=e.length-4,n=3,r,i;if((e[n][1].type==="lineEnding"||e[n][1].type==="space")&&(e[t][1].type==="lineEnding"||e[t][1].type==="space")){for(r=n;++r<t;)if(e[r][1].type==="codeTextData"){e[n][1].type="codeTextPadding",e[t][1].type="codeTextPadding",n+=2,t-=2;break}}for(r=n-1,t++;++r<=t;)i===void 0?r!==t&&e[r][1].type!=="lineEnding"&&(i=r):(r===t||e[r][1].type==="lineEnding")&&(e[i][1].type="codeTextData",r!==i+2&&(e[i][1].end=e[r-1][1].end,e.splice(i+2,r-i-2),t-=r-i-2,r=i+2),i=void 0);return e}function ww(e){return e!==96||this.events[this.events.length-1][1].type==="characterEscape"}function bw(e,t,n){let r=0,i,a;return o;function o(c){return e.enter("codeText"),e.enter("codeTextSequence"),s(c)}function s(c){return c===96?(e.consume(c),r++,s):(e.exit("codeTextSequence"),l(c))}function l(c){return c===null?n(c):c===32?(e.enter("space"),e.consume(c),e.exit("space"),l):c===96?(a=e.enter("codeTextSequence"),i=0,d(c)):K(c)?(e.enter("lineEnding"),e.consume(c),e.exit("lineEnding"),l):(e.enter("codeTextData"),u(c))}function u(c){return c===null||c===32||c===96||K(c)?(e.exit("codeTextData"),l(c)):(e.consume(c),u)}function d(c){return c===96?(e.consume(c),i++,d):i===r?(e.exit("codeTextSequence"),e.exit("codeText"),t(c)):(a.type="codeTextData",u(c))}}class kw{constructor(t){this.left=t?[...t]:[],this.right=[]}get(t){if(t<0||t>=this.left.length+this.right.length)throw new RangeError("Cannot access index `"+t+"` in a splice buffer of size `"+(this.left.length+this.right.length)+"`");return t<this.left.length?this.left[t]:this.right[this.right.length-t+this.left.length-1]}get length(){return this.left.length+this.right.length}shift(){return this.setCursor(0),this.right.pop()}slice(t,n){const r=n??Number.POSITIVE_INFINITY;return r<this.left.length?this.left.slice(t,r):t>this.left.length?this.right.slice(this.right.length-r+this.left.length,this.right.length-t+this.left.length).reverse():this.left.slice(t).concat(this.right.slice(this.right.length-r+this.left.length).reverse())}splice(t,n,r){const i=n||0;this.setCursor(Math.trunc(t));const a=this.right.splice(this.right.length-i,Number.POSITIVE_INFINITY);return r&&ii(this.left,r),a.reverse()}pop(){return this.setCursor(Number.POSITIVE_INFINITY),this.left.pop()}push(t){this.setCursor(Number.POSITIVE_INFINITY),this.left.push(t)}pushMany(t){this.setCursor(Number.POSITIVE_INFINITY),ii(this.left,t)}unshift(t){this.setCursor(0),this.right.push(t)}unshiftMany(t){this.setCursor(0),ii(this.right,t.reverse())}setCursor(t){if(!(t===this.left.length||t>this.left.length&&this.right.length===0||t<0&&this.left.length===0))if(t<this.left.length){const n=this.left.splice(t,Number.POSITIVE_INFINITY);ii(this.right,n.reverse())}else{const n=this.right.splice(this.left.length+this.right.length-t,Number.POSITIVE_INFINITY);ii(this.left,n.reverse())}}}function ii(e,t){let n=0;if(t.length<1e4)e.push(...t);else for(;n<t.length;)e.push(...t.slice(n,n+1e4)),n+=1e4}function sg(e){const t={};let n=-1,r,i,a,o,s,l,u;const d=new kw(e);for(;++n<d.length;){for(;n in t;)n=t[n];if(r=d.get(n),n&&r[1].type==="chunkFlow"&&d.get(n-1)[1].type==="listItemPrefix"&&(l=r[1]._tokenizer.events,a=0,a<l.length&&l[a][1].type==="lineEndingBlank"&&(a+=2),a<l.length&&l[a][1].type==="content"))for(;++a<l.length&&l[a][1].type!=="content";)l[a][1].type==="chunkText"&&(l[a][1]._isInFirstContentOfListItem=!0,a++);if(r[0]==="enter")r[1].contentType&&(Object.assign(t,Sw(d,n)),n=t[n],u=!0);else if(r[1]._container){for(a=n,i=void 0;a--;)if(o=d.get(a),o[1].type==="lineEnding"||o[1].type==="lineEndingBlank")o[0]==="enter"&&(i&&(d.get(i)[1].type="lineEndingBlank"),o[1].type="lineEnding",i=a);else if(!(o[1].type==="linePrefix"||o[1].type==="listItemIndent"))break;i&&(r[1].end={...d.get(i)[1].start},s=d.slice(i,n),s.unshift(r),d.splice(i,n-i+1,s))}}return st(e,0,Number.POSITIVE_INFINITY,d.slice(0)),!u}function Sw(e,t){const n=e.get(t)[1],r=e.get(t)[2];let i=t-1;const a=[];let o=n._tokenizer;o||(o=r.parser[n.contentType](n.start),n._contentTypeTextTrailing&&(o._contentTypeTextTrailing=!0));const s=o.events,l=[],u={};let d,c,p=-1,f=n,y=0,v=0;const k=[v];for(;f;){for(;e.get(++i)[1]!==f;);a.push(i),f._tokenizer||(d=r.sliceStream(f),f.next||d.push(null),c&&o.defineSkip(f.start),f._isInFirstContentOfListItem&&(o._gfmTasklistFirstContentOfListItem=!0),o.write(d),f._isInFirstContentOfListItem&&(o._gfmTasklistFirstContentOfListItem=void 0)),c=f,f=f.next}for(f=n;++p<s.length;)s[p][0]==="exit"&&s[p-1][0]==="enter"&&s[p][1].type===s[p-1][1].type&&s[p][1].start.line!==s[p][1].end.line&&(v=p+1,k.push(v),f._tokenizer=void 0,f.previous=void 0,f=f.next);for(o.events=[],f?(f._tokenizer=void 0,f.previous=void 0):k.pop(),p=k.length;p--;){const m=s.slice(k[p],k[p+1]),g=a.pop();l.push([g,g+m.length-1]),e.splice(g,2,m)}for(l.reverse(),p=-1;++p<l.length;)u[y+l[p][0]]=y+l[p][1],y+=l[p][1]-l[p][0]-1;return u}const Cw={resolve:zw,tokenize:Tw},Ew={partial:!0,tokenize:Pw};function zw(e){return sg(e),e}function Tw(e,t){let n;return r;function r(s){return e.enter("content"),n=e.enter("chunkContent",{contentType:"content"}),i(s)}function i(s){return s===null?a(s):K(s)?e.check(Ew,o,a)(s):(e.consume(s),i)}function a(s){return e.exit("chunkContent"),e.exit("content"),t(s)}function o(s){return e.consume(s),e.exit("chunkContent"),n.next=e.enter("chunkContent",{contentType:"content",previous:n}),n=n.next,i}}function Pw(e,t,n){const r=this;return i;function i(o){return e.exit("chunkContent"),e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),re(e,a,"linePrefix")}function a(o){if(o===null||K(o))return n(o);const s=r.events[r.events.length-1];return!r.parser.constructs.disable.null.includes("codeIndented")&&s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(o):e.interrupt(r.parser.constructs.flow,n,t)(o)}}function lg(e,t,n,r,i,a,o,s,l){const u=l||Number.POSITIVE_INFINITY;let d=0;return c;function c(m){return m===60?(e.enter(r),e.enter(i),e.enter(a),e.consume(m),e.exit(a),p):m===null||m===32||m===41||ko(m)?n(m):(e.enter(r),e.enter(o),e.enter(s),e.enter("chunkString",{contentType:"string"}),v(m))}function p(m){return m===62?(e.enter(a),e.consume(m),e.exit(a),e.exit(i),e.exit(r),t):(e.enter(s),e.enter("chunkString",{contentType:"string"}),f(m))}function f(m){return m===62?(e.exit("chunkString"),e.exit(s),p(m)):m===null||m===60||K(m)?n(m):(e.consume(m),m===92?y:f)}function y(m){return m===60||m===62||m===92?(e.consume(m),f):f(m)}function v(m){return!d&&(m===null||m===41||ce(m))?(e.exit("chunkString"),e.exit(s),e.exit(o),e.exit(r),t(m)):d<u&&m===40?(e.consume(m),d++,v):m===41?(e.consume(m),d--,v):m===null||m===32||m===40||ko(m)?n(m):(e.consume(m),m===92?k:v)}function k(m){return m===40||m===41||m===92?(e.consume(m),v):v(m)}}function ug(e,t,n,r,i,a){const o=this;let s=0,l;return u;function u(f){return e.enter(r),e.enter(i),e.consume(f),e.exit(i),e.enter(a),d}function d(f){return s>999||f===null||f===91||f===93&&!l||f===94&&!s&&"_hiddenFootnoteSupport"in o.parser.constructs?n(f):f===93?(e.exit(a),e.enter(i),e.consume(f),e.exit(i),e.exit(r),t):K(f)?(e.enter("lineEnding"),e.consume(f),e.exit("lineEnding"),d):(e.enter("chunkString",{contentType:"string"}),c(f))}function c(f){return f===null||f===91||f===93||K(f)||s++>999?(e.exit("chunkString"),d(f)):(e.consume(f),l||(l=!Z(f)),f===92?p:c)}function p(f){return f===91||f===92||f===93?(e.consume(f),s++,c):c(f)}}function cg(e,t,n,r,i,a){let o;return s;function s(p){return p===34||p===39||p===40?(e.enter(r),e.enter(i),e.consume(p),e.exit(i),o=p===40?41:p,l):n(p)}function l(p){return p===o?(e.enter(i),e.consume(p),e.exit(i),e.exit(r),t):(e.enter(a),u(p))}function u(p){return p===o?(e.exit(a),l(o)):p===null?n(p):K(p)?(e.enter("lineEnding"),e.consume(p),e.exit("lineEnding"),re(e,u,"linePrefix")):(e.enter("chunkString",{contentType:"string"}),d(p))}function d(p){return p===o||p===null||K(p)?(e.exit("chunkString"),u(p)):(e.consume(p),p===92?c:d)}function c(p){return p===o||p===92?(e.consume(p),d):d(p)}}function ki(e,t){let n;return r;function r(i){return K(i)?(e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),n=!0,r):Z(i)?re(e,r,n?"linePrefix":"lineSuffix")(i):t(i)}}const Nw={name:"definition",tokenize:Lw},Aw={partial:!0,tokenize:jw};function Lw(e,t,n){const r=this;let i;return a;function a(f){return e.enter("definition"),o(f)}function o(f){return ug.call(r,e,s,n,"definitionLabel","definitionLabelMarker","definitionLabelString")(f)}function s(f){return i=Rt(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)),f===58?(e.enter("definitionMarker"),e.consume(f),e.exit("definitionMarker"),l):n(f)}function l(f){return ce(f)?ki(e,u)(f):u(f)}function u(f){return lg(e,d,n,"definitionDestination","definitionDestinationLiteral","definitionDestinationLiteralMarker","definitionDestinationRaw","definitionDestinationString")(f)}function d(f){return e.attempt(Aw,c,c)(f)}function c(f){return Z(f)?re(e,p,"whitespace")(f):p(f)}function p(f){return f===null||K(f)?(e.exit("definition"),r.parser.defined.push(i),t(f)):n(f)}}function jw(e,t,n){return r;function r(s){return ce(s)?ki(e,i)(s):n(s)}function i(s){return cg(e,a,n,"definitionTitle","definitionTitleMarker","definitionTitleString")(s)}function a(s){return Z(s)?re(e,o,"whitespace")(s):o(s)}function o(s){return s===null||K(s)?t(s):n(s)}}const Rw={name:"hardBreakEscape",tokenize:Iw};function Iw(e,t,n){return r;function r(a){return e.enter("hardBreakEscape"),e.consume(a),i}function i(a){return K(a)?(e.exit("hardBreakEscape"),t(a)):n(a)}}const Ow={name:"headingAtx",resolve:Dw,tokenize:Mw};function Dw(e,t){let n=e.length-2,r=3,i,a;return e[r][1].type==="whitespace"&&(r+=2),n-2>r&&e[n][1].type==="whitespace"&&(n-=2),e[n][1].type==="atxHeadingSequence"&&(r===n-1||n-4>r&&e[n-2][1].type==="whitespace")&&(n-=r+1===n?2:4),n>r&&(i={type:"atxHeadingText",start:e[r][1].start,end:e[n][1].end},a={type:"chunkText",start:e[r][1].start,end:e[n][1].end,contentType:"text"},st(e,r,n-r+1,[["enter",i,t],["enter",a,t],["exit",a,t],["exit",i,t]])),e}function Mw(e,t,n){let r=0;return i;function i(d){return e.enter("atxHeading"),a(d)}function a(d){return e.enter("atxHeadingSequence"),o(d)}function o(d){return d===35&&r++<6?(e.consume(d),o):d===null||ce(d)?(e.exit("atxHeadingSequence"),s(d)):n(d)}function s(d){return d===35?(e.enter("atxHeadingSequence"),l(d)):d===null||K(d)?(e.exit("atxHeading"),t(d)):Z(d)?re(e,s,"whitespace")(d):(e.enter("atxHeadingText"),u(d))}function l(d){return d===35?(e.consume(d),l):(e.exit("atxHeadingSequence"),s(d))}function u(d){return d===null||d===35||ce(d)?(e.exit("atxHeadingText"),s(d)):(e.consume(d),u)}}const Fw=["address","article","aside","base","basefont","blockquote","body","caption","center","col","colgroup","dd","details","dialog","dir","div","dl","dt","fieldset","figcaption","figure","footer","form","frame","frameset","h1","h2","h3","h4","h5","h6","head","header","hr","html","iframe","legend","li","link","main","menu","menuitem","nav","noframes","ol","optgroup","option","p","param","search","section","summary","table","tbody","td","tfoot","th","thead","title","tr","track","ul"],_p=["pre","script","style","textarea"],qw={concrete:!0,name:"htmlFlow",resolveTo:Uw,tokenize:Vw},Ww={partial:!0,tokenize:$w},Bw={partial:!0,tokenize:Hw};function Uw(e){let t=e.length;for(;t--&&!(e[t][0]==="enter"&&e[t][1].type==="htmlFlow"););return t>1&&e[t-2][1].type==="linePrefix"&&(e[t][1].start=e[t-2][1].start,e[t+1][1].start=e[t-2][1].start,e.splice(t-2,2)),e}function Vw(e,t,n){const r=this;let i,a,o,s,l;return u;function u(w){return d(w)}function d(w){return e.enter("htmlFlow"),e.enter("htmlFlowData"),e.consume(w),c}function c(w){return w===33?(e.consume(w),p):w===47?(e.consume(w),a=!0,v):w===63?(e.consume(w),i=3,r.interrupt?t:_):Ve(w)?(e.consume(w),o=String.fromCharCode(w),k):n(w)}function p(w){return w===45?(e.consume(w),i=2,f):w===91?(e.consume(w),i=5,s=0,y):Ve(w)?(e.consume(w),i=4,r.interrupt?t:_):n(w)}function f(w){return w===45?(e.consume(w),r.interrupt?t:_):n(w)}function y(w){const ie="CDATA[";return w===ie.charCodeAt(s++)?(e.consume(w),s===ie.length?r.interrupt?t:W:y):n(w)}function v(w){return Ve(w)?(e.consume(w),o=String.fromCharCode(w),k):n(w)}function k(w){if(w===null||w===47||w===62||ce(w)){const ie=w===47,be=o.toLowerCase();return!ie&&!a&&_p.includes(be)?(i=1,r.interrupt?t(w):W(w)):Fw.includes(o.toLowerCase())?(i=6,ie?(e.consume(w),m):r.interrupt?t(w):W(w)):(i=7,r.interrupt&&!r.parser.lazy[r.now().line]?n(w):a?g(w):x(w))}return w===45||Fe(w)?(e.consume(w),o+=String.fromCharCode(w),k):n(w)}function m(w){return w===62?(e.consume(w),r.interrupt?t:W):n(w)}function g(w){return Z(w)?(e.consume(w),g):b(w)}function x(w){return w===47?(e.consume(w),b):w===58||w===95||Ve(w)?(e.consume(w),C):Z(w)?(e.consume(w),x):b(w)}function C(w){return w===45||w===46||w===58||w===95||Fe(w)?(e.consume(w),C):z(w)}function z(w){return w===61?(e.consume(w),S):Z(w)?(e.consume(w),z):x(w)}function S(w){return w===null||w===60||w===61||w===62||w===96?n(w):w===34||w===39?(e.consume(w),l=w,T):Z(w)?(e.consume(w),S):N(w)}function T(w){return w===l?(e.consume(w),l=null,F):w===null||K(w)?n(w):(e.consume(w),T)}function N(w){return w===null||w===34||w===39||w===47||w===60||w===61||w===62||w===96||ce(w)?z(w):(e.consume(w),N)}function F(w){return w===47||w===62||Z(w)?x(w):n(w)}function b(w){return w===62?(e.consume(w),I):n(w)}function I(w){return w===null||K(w)?W(w):Z(w)?(e.consume(w),I):n(w)}function W(w){return w===45&&i===2?(e.consume(w),$):w===60&&i===1?(e.consume(w),ne):w===62&&i===4?(e.consume(w),B):w===63&&i===3?(e.consume(w),_):w===93&&i===5?(e.consume(w),V):K(w)&&(i===6||i===7)?(e.exit("htmlFlowData"),e.check(Ww,Q,X)(w)):w===null||K(w)?(e.exit("htmlFlowData"),X(w)):(e.consume(w),W)}function X(w){return e.check(Bw,ee,Q)(w)}function ee(w){return e.enter("lineEnding"),e.consume(w),e.exit("lineEnding"),A}function A(w){return w===null||K(w)?X(w):(e.enter("htmlFlowData"),W(w))}function $(w){return w===45?(e.consume(w),_):W(w)}function ne(w){return w===47?(e.consume(w),o="",O):W(w)}function O(w){if(w===62){const ie=o.toLowerCase();return _p.includes(ie)?(e.consume(w),B):W(w)}return Ve(w)&&o.length<8?(e.consume(w),o+=String.fromCharCode(w),O):W(w)}function V(w){return w===93?(e.consume(w),_):W(w)}function _(w){return w===62?(e.consume(w),B):w===45&&i===2?(e.consume(w),_):W(w)}function B(w){return w===null||K(w)?(e.exit("htmlFlowData"),Q(w)):(e.consume(w),B)}function Q(w){return e.exit("htmlFlow"),t(w)}}function Hw(e,t,n){const r=this;return i;function i(o){return K(o)?(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),a):n(o)}function a(o){return r.parser.lazy[r.now().line]?n(o):t(o)}}function $w(e,t,n){return r;function r(i){return e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),e.attempt(ia,t,n)}}const Qw={name:"htmlText",tokenize:Kw};function Kw(e,t,n){const r=this;let i,a,o;return s;function s(_){return e.enter("htmlText"),e.enter("htmlTextData"),e.consume(_),l}function l(_){return _===33?(e.consume(_),u):_===47?(e.consume(_),z):_===63?(e.consume(_),x):Ve(_)?(e.consume(_),N):n(_)}function u(_){return _===45?(e.consume(_),d):_===91?(e.consume(_),a=0,y):Ve(_)?(e.consume(_),g):n(_)}function d(_){return _===45?(e.consume(_),f):n(_)}function c(_){return _===null?n(_):_===45?(e.consume(_),p):K(_)?(o=c,ne(_)):(e.consume(_),c)}function p(_){return _===45?(e.consume(_),f):c(_)}function f(_){return _===62?$(_):_===45?p(_):c(_)}function y(_){const B="CDATA[";return _===B.charCodeAt(a++)?(e.consume(_),a===B.length?v:y):n(_)}function v(_){return _===null?n(_):_===93?(e.consume(_),k):K(_)?(o=v,ne(_)):(e.consume(_),v)}function k(_){return _===93?(e.consume(_),m):v(_)}function m(_){return _===62?$(_):_===93?(e.consume(_),m):v(_)}function g(_){return _===null||_===62?$(_):K(_)?(o=g,ne(_)):(e.consume(_),g)}function x(_){return _===null?n(_):_===63?(e.consume(_),C):K(_)?(o=x,ne(_)):(e.consume(_),x)}function C(_){return _===62?$(_):x(_)}function z(_){return Ve(_)?(e.consume(_),S):n(_)}function S(_){return _===45||Fe(_)?(e.consume(_),S):T(_)}function T(_){return K(_)?(o=T,ne(_)):Z(_)?(e.consume(_),T):$(_)}function N(_){return _===45||Fe(_)?(e.consume(_),N):_===47||_===62||ce(_)?F(_):n(_)}function F(_){return _===47?(e.consume(_),$):_===58||_===95||Ve(_)?(e.consume(_),b):K(_)?(o=F,ne(_)):Z(_)?(e.consume(_),F):$(_)}function b(_){return _===45||_===46||_===58||_===95||Fe(_)?(e.consume(_),b):I(_)}function I(_){return _===61?(e.consume(_),W):K(_)?(o=I,ne(_)):Z(_)?(e.consume(_),I):F(_)}function W(_){return _===null||_===60||_===61||_===62||_===96?n(_):_===34||_===39?(e.consume(_),i=_,X):K(_)?(o=W,ne(_)):Z(_)?(e.consume(_),W):(e.consume(_),ee)}function X(_){return _===i?(e.consume(_),i=void 0,A):_===null?n(_):K(_)?(o=X,ne(_)):(e.consume(_),X)}function ee(_){return _===null||_===34||_===39||_===60||_===61||_===96?n(_):_===47||_===62||ce(_)?F(_):(e.consume(_),ee)}function A(_){return _===47||_===62||ce(_)?F(_):n(_)}function $(_){return _===62?(e.consume(_),e.exit("htmlTextData"),e.exit("htmlText"),t):n(_)}function ne(_){return e.exit("htmlTextData"),e.enter("lineEnding"),e.consume(_),e.exit("lineEnding"),O}function O(_){return Z(_)?re(e,V,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(_):V(_)}function V(_){return e.enter("htmlTextData"),o(_)}}const _c={name:"labelEnd",resolveAll:Zw,resolveTo:Jw,tokenize:eb},Xw={tokenize:tb},Yw={tokenize:nb},Gw={tokenize:rb};function Zw(e){let t=-1;const n=[];for(;++t<e.length;){const r=e[t][1];if(n.push(e[t]),r.type==="labelImage"||r.type==="labelLink"||r.type==="labelEnd"){const i=r.type==="labelImage"?4:2;r.type="data",t+=i}}return e.length!==n.length&&st(e,0,e.length,n),e}function Jw(e,t){let n=e.length,r=0,i,a,o,s;for(;n--;)if(i=e[n][1],a){if(i.type==="link"||i.type==="labelLink"&&i._inactive)break;e[n][0]==="enter"&&i.type==="labelLink"&&(i._inactive=!0)}else if(o){if(e[n][0]==="enter"&&(i.type==="labelImage"||i.type==="labelLink")&&!i._balanced&&(a=n,i.type!=="labelLink")){r=2;break}}else i.type==="labelEnd"&&(o=n);const l={type:e[a][1].type==="labelLink"?"link":"image",start:{...e[a][1].start},end:{...e[e.length-1][1].end}},u={type:"label",start:{...e[a][1].start},end:{...e[o][1].end}},d={type:"labelText",start:{...e[a+r+2][1].end},end:{...e[o-2][1].start}};return s=[["enter",l,t],["enter",u,t]],s=gt(s,e.slice(a+1,a+r+3)),s=gt(s,[["enter",d,t]]),s=gt(s,Yo(t.parser.constructs.insideSpan.null,e.slice(a+r+4,o-3),t)),s=gt(s,[["exit",d,t],e[o-2],e[o-1],["exit",u,t]]),s=gt(s,e.slice(o+1)),s=gt(s,[["exit",l,t]]),st(e,a,e.length,s),e}function eb(e,t,n){const r=this;let i=r.events.length,a,o;for(;i--;)if((r.events[i][1].type==="labelImage"||r.events[i][1].type==="labelLink")&&!r.events[i][1]._balanced){a=r.events[i][1];break}return s;function s(p){return a?a._inactive?c(p):(o=r.parser.defined.includes(Rt(r.sliceSerialize({start:a.end,end:r.now()}))),e.enter("labelEnd"),e.enter("labelMarker"),e.consume(p),e.exit("labelMarker"),e.exit("labelEnd"),l):n(p)}function l(p){return p===40?e.attempt(Xw,d,o?d:c)(p):p===91?e.attempt(Yw,d,o?u:c)(p):o?d(p):c(p)}function u(p){return e.attempt(Gw,d,c)(p)}function d(p){return t(p)}function c(p){return a._balanced=!0,n(p)}}function tb(e,t,n){return r;function r(c){return e.enter("resource"),e.enter("resourceMarker"),e.consume(c),e.exit("resourceMarker"),i}function i(c){return ce(c)?ki(e,a)(c):a(c)}function a(c){return c===41?d(c):lg(e,o,s,"resourceDestination","resourceDestinationLiteral","resourceDestinationLiteralMarker","resourceDestinationRaw","resourceDestinationString",32)(c)}function o(c){return ce(c)?ki(e,l)(c):d(c)}function s(c){return n(c)}function l(c){return c===34||c===39||c===40?cg(e,u,n,"resourceTitle","resourceTitleMarker","resourceTitleString")(c):d(c)}function u(c){return ce(c)?ki(e,d)(c):d(c)}function d(c){return c===41?(e.enter("resourceMarker"),e.consume(c),e.exit("resourceMarker"),e.exit("resource"),t):n(c)}}function nb(e,t,n){const r=this;return i;function i(s){return ug.call(r,e,a,o,"reference","referenceMarker","referenceString")(s)}function a(s){return r.parser.defined.includes(Rt(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)))?t(s):n(s)}function o(s){return n(s)}}function rb(e,t,n){return r;function r(a){return e.enter("reference"),e.enter("referenceMarker"),e.consume(a),e.exit("referenceMarker"),i}function i(a){return a===93?(e.enter("referenceMarker"),e.consume(a),e.exit("referenceMarker"),e.exit("reference"),t):n(a)}}const ib={name:"labelStartImage",resolveAll:_c.resolveAll,tokenize:ab};function ab(e,t,n){const r=this;return i;function i(s){return e.enter("labelImage"),e.enter("labelImageMarker"),e.consume(s),e.exit("labelImageMarker"),a}function a(s){return s===91?(e.enter("labelMarker"),e.consume(s),e.exit("labelMarker"),e.exit("labelImage"),o):n(s)}function o(s){return s===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(s):t(s)}}const ob={name:"labelStartLink",resolveAll:_c.resolveAll,tokenize:sb};function sb(e,t,n){const r=this;return i;function i(o){return e.enter("labelLink"),e.enter("labelMarker"),e.consume(o),e.exit("labelMarker"),e.exit("labelLink"),a}function a(o){return o===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(o):t(o)}}const Os={name:"lineEnding",tokenize:lb};function lb(e,t){return n;function n(r){return e.enter("lineEnding"),e.consume(r),e.exit("lineEnding"),re(e,t,"linePrefix")}}const Qa={name:"thematicBreak",tokenize:ub};function ub(e,t,n){let r=0,i;return a;function a(u){return e.enter("thematicBreak"),o(u)}function o(u){return i=u,s(u)}function s(u){return u===i?(e.enter("thematicBreakSequence"),l(u)):r>=3&&(u===null||K(u))?(e.exit("thematicBreak"),t(u)):n(u)}function l(u){return u===i?(e.consume(u),r++,l):(e.exit("thematicBreakSequence"),Z(u)?re(e,s,"whitespace")(u):s(u))}}const Ke={continuation:{tokenize:fb},exit:hb,name:"list",tokenize:pb},cb={partial:!0,tokenize:gb},db={partial:!0,tokenize:mb};function pb(e,t,n){const r=this,i=r.events[r.events.length-1];let a=i&&i[1].type==="linePrefix"?i[2].sliceSerialize(i[1],!0).length:0,o=0;return s;function s(f){const y=r.containerState.type||(f===42||f===43||f===45?"listUnordered":"listOrdered");if(y==="listUnordered"?!r.containerState.marker||f===r.containerState.marker:ou(f)){if(r.containerState.type||(r.containerState.type=y,e.enter(y,{_container:!0})),y==="listUnordered")return e.enter("listItemPrefix"),f===42||f===45?e.check(Qa,n,u)(f):u(f);if(!r.interrupt||f===49)return e.enter("listItemPrefix"),e.enter("listItemValue"),l(f)}return n(f)}function l(f){return ou(f)&&++o<10?(e.consume(f),l):(!r.interrupt||o<2)&&(r.containerState.marker?f===r.containerState.marker:f===41||f===46)?(e.exit("listItemValue"),u(f)):n(f)}function u(f){return e.enter("listItemMarker"),e.consume(f),e.exit("listItemMarker"),r.containerState.marker=r.containerState.marker||f,e.check(ia,r.interrupt?n:d,e.attempt(cb,p,c))}function d(f){return r.containerState.initialBlankLine=!0,a++,p(f)}function c(f){return Z(f)?(e.enter("listItemPrefixWhitespace"),e.consume(f),e.exit("listItemPrefixWhitespace"),p):n(f)}function p(f){return r.containerState.size=a+r.sliceSerialize(e.exit("listItemPrefix"),!0).length,t(f)}}function fb(e,t,n){const r=this;return r.containerState._closeFlow=void 0,e.check(ia,i,a);function i(s){return r.containerState.furtherBlankLines=r.containerState.furtherBlankLines||r.containerState.initialBlankLine,re(e,t,"listItemIndent",r.containerState.size+1)(s)}function a(s){return r.containerState.furtherBlankLines||!Z(s)?(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,o(s)):(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,e.attempt(db,t,o)(s))}function o(s){return r.containerState._closeFlow=!0,r.interrupt=void 0,re(e,e.attempt(Ke,t,n),"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(s)}}function mb(e,t,n){const r=this;return re(e,i,"listItemIndent",r.containerState.size+1);function i(a){const o=r.events[r.events.length-1];return o&&o[1].type==="listItemIndent"&&o[2].sliceSerialize(o[1],!0).length===r.containerState.size?t(a):n(a)}}function hb(e){e.exit(this.containerState.type)}function gb(e,t,n){const r=this;return re(e,i,"listItemPrefixWhitespace",r.parser.constructs.disable.null.includes("codeIndented")?void 0:5);function i(a){const o=r.events[r.events.length-1];return!Z(a)&&o&&o[1].type==="listItemPrefixWhitespace"?t(a):n(a)}}const wp={name:"setextUnderline",resolveTo:yb,tokenize:xb};function yb(e,t){let n=e.length,r,i,a;for(;n--;)if(e[n][0]==="enter"){if(e[n][1].type==="content"){r=n;break}e[n][1].type==="paragraph"&&(i=n)}else e[n][1].type==="content"&&e.splice(n,1),!a&&e[n][1].type==="definition"&&(a=n);const o={type:"setextHeading",start:{...e[r][1].start},end:{...e[e.length-1][1].end}};return e[i][1].type="setextHeadingText",a?(e.splice(i,0,["enter",o,t]),e.splice(a+1,0,["exit",e[r][1],t]),e[r][1].end={...e[a][1].end}):e[r][1]=o,e.push(["exit",o,t]),e}function xb(e,t,n){const r=this;let i;return a;function a(u){let d=r.events.length,c;for(;d--;)if(r.events[d][1].type!=="lineEnding"&&r.events[d][1].type!=="linePrefix"&&r.events[d][1].type!=="content"){c=r.events[d][1].type==="paragraph";break}return!r.parser.lazy[r.now().line]&&(r.interrupt||c)?(e.enter("setextHeadingLine"),i=u,o(u)):n(u)}function o(u){return e.enter("setextHeadingLineSequence"),s(u)}function s(u){return u===i?(e.consume(u),s):(e.exit("setextHeadingLineSequence"),Z(u)?re(e,l,"lineSuffix")(u):l(u))}function l(u){return u===null||K(u)?(e.exit("setextHeadingLine"),t(u)):n(u)}}const vb={tokenize:_b};function _b(e){const t=this,n=e.attempt(ia,r,e.attempt(this.parser.constructs.flowInitial,i,re(e,e.attempt(this.parser.constructs.flow,i,e.attempt(Cw,i)),"linePrefix")));return n;function r(a){if(a===null){e.consume(a);return}return e.enter("lineEndingBlank"),e.consume(a),e.exit("lineEndingBlank"),t.currentConstruct=void 0,n}function i(a){if(a===null){e.consume(a);return}return e.enter("lineEnding"),e.consume(a),e.exit("lineEnding"),t.currentConstruct=void 0,n}}const wb={resolveAll:pg()},bb=dg("string"),kb=dg("text");function dg(e){return{resolveAll:pg(e==="text"?Sb:void 0),tokenize:t};function t(n){const r=this,i=this.parser.constructs[e],a=n.attempt(i,o,s);return o;function o(d){return u(d)?a(d):s(d)}function s(d){if(d===null){n.consume(d);return}return n.enter("data"),n.consume(d),l}function l(d){return u(d)?(n.exit("data"),a(d)):(n.consume(d),l)}function u(d){if(d===null)return!0;const c=i[d];let p=-1;if(c)for(;++p<c.length;){const f=c[p];if(!f.previous||f.previous.call(r,r.previous))return!0}return!1}}}function pg(e){return t;function t(n,r){let i=-1,a;for(;++i<=n.length;)a===void 0?n[i]&&n[i][1].type==="data"&&(a=i,i++):(!n[i]||n[i][1].type!=="data")&&(i!==a+2&&(n[a][1].end=n[i-1][1].end,n.splice(a+2,i-a-2),i=a+2),a=void 0);return e?e(n,r):n}}function Sb(e,t){let n=0;for(;++n<=e.length;)if((n===e.length||e[n][1].type==="lineEnding")&&e[n-1][1].type==="data"){const r=e[n-1][1],i=t.sliceStream(r);let a=i.length,o=-1,s=0,l;for(;a--;){const u=i[a];if(typeof u=="string"){for(o=u.length;u.charCodeAt(o-1)===32;)s++,o--;if(o)break;o=-1}else if(u===-2)l=!0,s++;else if(u!==-1){a++;break}}if(t._contentTypeTextTrailing&&n===e.length&&(s=0),s){const u={type:n===e.length||l||s<2?"lineSuffix":"hardBreakTrailing",start:{_bufferIndex:a?o:r.start._bufferIndex+o,_index:r.start._index+a,line:r.end.line,column:r.end.column-s,offset:r.end.offset-s},end:{...r.end}};r.end={...u.start},r.start.offset===r.end.offset?Object.assign(r,u):(e.splice(n,0,["enter",u,t],["exit",u,t]),n+=2)}n++}return e}const Cb={42:Ke,43:Ke,45:Ke,48:Ke,49:Ke,50:Ke,51:Ke,52:Ke,53:Ke,54:Ke,55:Ke,56:Ke,57:Ke,62:ig},Eb={91:Nw},zb={[-2]:Is,[-1]:Is,32:Is},Tb={35:Ow,42:Qa,45:[wp,Qa],60:qw,61:wp,95:Qa,96:vp,126:vp},Pb={38:og,92:ag},Nb={[-5]:Os,[-4]:Os,[-3]:Os,33:ib,38:og,42:su,60:[ow,Qw],91:ob,92:[Rw,ag],93:_c,95:su,96:vw},Ab={null:[su,wb]},Lb={null:[42,95]},jb={null:[]},Rb=Object.freeze(Object.defineProperty({__proto__:null,attentionMarkers:Lb,contentInitial:Eb,disable:jb,document:Cb,flow:Tb,flowInitial:zb,insideSpan:Ab,string:Pb,text:Nb},Symbol.toStringTag,{value:"Module"}));function Ib(e,t,n){let r={_bufferIndex:-1,_index:0,line:n&&n.line||1,column:n&&n.column||1,offset:n&&n.offset||0};const i={},a=[];let o=[],s=[];const l={attempt:T(z),check:T(S),consume:g,enter:x,exit:C,interrupt:T(S,{interrupt:!0})},u={code:null,containerState:{},defineSkip:v,events:[],now:y,parser:e,previous:null,sliceSerialize:p,sliceStream:f,write:c};let d=t.tokenize.call(u,l);return t.resolveAll&&a.push(t),u;function c(I){return o=gt(o,I),k(),o[o.length-1]!==null?[]:(N(t,0),u.events=Yo(a,u.events,u),u.events)}function p(I,W){return Db(f(I),W)}function f(I){return Ob(o,I)}function y(){const{_bufferIndex:I,_index:W,line:X,column:ee,offset:A}=r;return{_bufferIndex:I,_index:W,line:X,column:ee,offset:A}}function v(I){i[I.line]=I.column,b()}function k(){let I;for(;r._index<o.length;){const W=o[r._index];if(typeof W=="string")for(I=r._index,r._bufferIndex<0&&(r._bufferIndex=0);r._index===I&&r._bufferIndex<W.length;)m(W.charCodeAt(r._bufferIndex));else m(W)}}function m(I){d=d(I)}function g(I){K(I)?(r.line++,r.column=1,r.offset+=I===-3?2:1,b()):I!==-1&&(r.column++,r.offset++),r._bufferIndex<0?r._index++:(r._bufferIndex++,r._bufferIndex===o[r._index].length&&(r._bufferIndex=-1,r._index++)),u.previous=I}function x(I,W){const X=W||{};return X.type=I,X.start=y(),u.events.push(["enter",X,u]),s.push(X),X}function C(I){const W=s.pop();return W.end=y(),u.events.push(["exit",W,u]),W}function z(I,W){N(I,W.from)}function S(I,W){W.restore()}function T(I,W){return X;function X(ee,A,$){let ne,O,V,_;return Array.isArray(ee)?Q(ee):"tokenize"in ee?Q([ee]):B(ee);function B(le){return dt;function dt(q){const L=q!==null&&le[q],D=q!==null&&le.null,H=[...Array.isArray(L)?L:L?[L]:[],...Array.isArray(D)?D:D?[D]:[]];return Q(H)(q)}}function Q(le){return ne=le,O=0,le.length===0?$:w(le[O])}function w(le){return dt;function dt(q){return _=F(),V=le,le.partial||(u.currentConstruct=le),le.name&&u.parser.constructs.disable.null.includes(le.name)?be():le.tokenize.call(W?Object.assign(Object.create(u),W):u,l,ie,be)(q)}}function ie(le){return I(V,_),A}function be(le){return _.restore(),++O<ne.length?w(ne[O]):$}}}function N(I,W){I.resolveAll&&!a.includes(I)&&a.push(I),I.resolve&&st(u.events,W,u.events.length-W,I.resolve(u.events.slice(W),u)),I.resolveTo&&(u.events=I.resolveTo(u.events,u))}function F(){const I=y(),W=u.previous,X=u.currentConstruct,ee=u.events.length,A=Array.from(s);return{from:ee,restore:$};function $(){r=I,u.previous=W,u.currentConstruct=X,u.events.length=ee,s=A,b()}}function b(){r.line in i&&r.column<2&&(r.column=i[r.line],r.offset+=i[r.line]-1)}}function Ob(e,t){const n=t.start._index,r=t.start._bufferIndex,i=t.end._index,a=t.end._bufferIndex;let o;if(n===i)o=[e[n].slice(r,a)];else{if(o=e.slice(n,i),r>-1){const s=o[0];typeof s=="string"?o[0]=s.slice(r):o.shift()}a>0&&o.push(e[i].slice(0,a))}return o}function Db(e,t){let n=-1;const r=[];let i;for(;++n<e.length;){const a=e[n];let o;if(typeof a=="string")o=a;else switch(a){case-5:{o="\r";break}case-4:{o=`
`;break}case-3:{o=`\r
`;break}case-2:{o=t?" ":"	";break}case-1:{if(!t&&i)continue;o=" ";break}default:o=String.fromCharCode(a)}i=a===-2,r.push(o)}return r.join("")}function Mb(e){const r={constructs:ng([Rb,...(e||{}).extensions||[]]),content:i(J_),defined:[],document:i(tw),flow:i(vb),lazy:{},string:i(bb),text:i(kb)};return r;function i(a){return o;function o(s){return Ib(r,a,s)}}}function Fb(e){for(;!sg(e););return e}const bp=/[\0\t\n\r]/g;function qb(){let e=1,t="",n=!0,r;return i;function i(a,o,s){const l=[];let u,d,c,p,f;for(a=t+(typeof a=="string"?a.toString():new TextDecoder(o||void 0).decode(a)),c=0,t="",n&&(a.charCodeAt(0)===65279&&c++,n=void 0);c<a.length;){if(bp.lastIndex=c,u=bp.exec(a),p=u&&u.index!==void 0?u.index:a.length,f=a.charCodeAt(p),!u){t=a.slice(c);break}if(f===10&&c===p&&r)l.push(-3),r=void 0;else switch(r&&(l.push(-5),r=void 0),c<p&&(l.push(a.slice(c,p)),e+=p-c),f){case 0:{l.push(65533),e++;break}case 9:{for(d=Math.ceil(e/4)*4,l.push(-2);e++<d;)l.push(-1);break}case 10:{l.push(-4),e=1;break}default:r=!0,e=1}c=p+1}return s&&(r&&l.push(-5),t&&l.push(t),l.push(null)),l}}const Wb=/\\([!-/:-@[-`{-~])|&(#(?:\d{1,7}|x[\da-f]{1,6})|[\da-z]{1,31});/gi;function Bb(e){return e.replace(Wb,Ub)}function Ub(e,t,n){if(t)return t;if(n.charCodeAt(0)===35){const i=n.charCodeAt(1),a=i===120||i===88;return rg(n.slice(a?2:1),a?16:10)}return vc(n)||e}const fg={}.hasOwnProperty;function Vb(e,t,n){return typeof t!="string"&&(n=t,t=void 0),Hb(n)(Fb(Mb(n).document().write(qb()(e,t,!0))))}function Hb(e){const t={transforms:[],canContainEols:["emphasis","fragment","heading","paragraph","strong"],enter:{autolink:a(Ac),autolinkProtocol:F,autolinkEmail:F,atxHeading:a(Ot),blockQuote:a(D),characterEscape:F,characterReference:F,codeFenced:a(H),codeFencedFenceInfo:o,codeFencedFenceMeta:o,codeIndented:a(H,o),codeText:a(Ee,o),codeTextData:F,data:F,codeFlowValue:F,definition:a(ve),definitionDestinationString:o,definitionLabelString:o,definitionTitleString:o,emphasis:a(Be),hardBreakEscape:a(jn),hardBreakTrailing:a(jn),htmlFlow:a(Rn,o),htmlFlowData:F,htmlText:a(Rn,o),htmlTextData:F,image:a(ly),label:o,link:a(Ac),listItem:a(uy),listItemValue:p,listOrdered:a(Lc,c),listUnordered:a(Lc),paragraph:a(cy),reference:w,referenceString:o,resourceDestinationString:o,resourceTitleString:o,setextHeading:a(Ot),strong:a(dy),thematicBreak:a(fy)},exit:{atxHeading:l(),atxHeadingSequence:z,autolink:l(),autolinkEmail:L,autolinkProtocol:q,blockQuote:l(),characterEscapeValue:b,characterReferenceMarkerHexadecimal:be,characterReferenceMarkerNumeric:be,characterReferenceValue:le,characterReference:dt,codeFenced:l(k),codeFencedFence:v,codeFencedFenceInfo:f,codeFencedFenceMeta:y,codeFlowValue:b,codeIndented:l(m),codeText:l(A),codeTextData:b,data:b,definition:l(),definitionDestinationString:C,definitionLabelString:g,definitionTitleString:x,emphasis:l(),hardBreakEscape:l(W),hardBreakTrailing:l(W),htmlFlow:l(X),htmlFlowData:b,htmlText:l(ee),htmlTextData:b,image:l(ne),label:V,labelText:O,lineEnding:I,link:l($),listItem:l(),listOrdered:l(),listUnordered:l(),paragraph:l(),referenceString:ie,resourceDestinationString:_,resourceTitleString:B,resource:Q,setextHeading:l(N),setextHeadingLineSequence:T,setextHeadingText:S,strong:l(),thematicBreak:l()}};mg(t,(e||{}).mdastExtensions||[]);const n={};return r;function r(P){let M={type:"root",children:[]};const Y={stack:[M],tokenStack:[],config:t,enter:s,exit:u,buffer:o,resume:d,data:n},te=[];let ue=-1;for(;++ue<P.length;)if(P[ue][1].type==="listOrdered"||P[ue][1].type==="listUnordered")if(P[ue][0]==="enter")te.push(ue);else{const wt=te.pop();ue=i(P,wt,ue)}for(ue=-1;++ue<P.length;){const wt=t[P[ue][0]];fg.call(wt,P[ue][1].type)&&wt[P[ue][1].type].call(Object.assign({sliceSerialize:P[ue][2].sliceSerialize},Y),P[ue][1])}if(Y.tokenStack.length>0){const wt=Y.tokenStack[Y.tokenStack.length-1];(wt[1]||kp).call(Y,void 0,wt[0])}for(M.position={start:sn(P.length>0?P[0][1].start:{line:1,column:1,offset:0}),end:sn(P.length>0?P[P.length-2][1].end:{line:1,column:1,offset:0})},ue=-1;++ue<t.transforms.length;)M=t.transforms[ue](M)||M;return M}function i(P,M,Y){let te=M-1,ue=-1,wt=!1,In,Vt,$r,Qr;for(;++te<=Y;){const tt=P[te];switch(tt[1].type){case"listUnordered":case"listOrdered":case"blockQuote":{tt[0]==="enter"?ue++:ue--,Qr=void 0;break}case"lineEndingBlank":{tt[0]==="enter"&&(In&&!Qr&&!ue&&!$r&&($r=te),Qr=void 0);break}case"linePrefix":case"listItemValue":case"listItemMarker":case"listItemPrefix":case"listItemPrefixWhitespace":break;default:Qr=void 0}if(!ue&&tt[0]==="enter"&&tt[1].type==="listItemPrefix"||ue===-1&&tt[0]==="exit"&&(tt[1].type==="listUnordered"||tt[1].type==="listOrdered")){if(In){let nr=te;for(Vt=void 0;nr--;){const Ht=P[nr];if(Ht[1].type==="lineEnding"||Ht[1].type==="lineEndingBlank"){if(Ht[0]==="exit")continue;Vt&&(P[Vt][1].type="lineEndingBlank",wt=!0),Ht[1].type="lineEnding",Vt=nr}else if(!(Ht[1].type==="linePrefix"||Ht[1].type==="blockQuotePrefix"||Ht[1].type==="blockQuotePrefixWhitespace"||Ht[1].type==="blockQuoteMarker"||Ht[1].type==="listItemIndent"))break}$r&&(!Vt||$r<Vt)&&(In._spread=!0),In.end=Object.assign({},Vt?P[Vt][1].start:tt[1].end),P.splice(Vt||te,0,["exit",In,tt[2]]),te++,Y++}if(tt[1].type==="listItemPrefix"){const nr={type:"listItem",_spread:!1,start:Object.assign({},tt[1].start),end:void 0};In=nr,P.splice(te,0,["enter",nr,tt[2]]),te++,Y++,$r=void 0,Qr=!0}}}return P[M][1]._spread=wt,Y}function a(P,M){return Y;function Y(te){s.call(this,P(te),te),M&&M.call(this,te)}}function o(){this.stack.push({type:"fragment",children:[]})}function s(P,M,Y){this.stack[this.stack.length-1].children.push(P),this.stack.push(P),this.tokenStack.push([M,Y||void 0]),P.position={start:sn(M.start),end:void 0}}function l(P){return M;function M(Y){P&&P.call(this,Y),u.call(this,Y)}}function u(P,M){const Y=this.stack.pop(),te=this.tokenStack.pop();if(te)te[0].type!==P.type&&(M?M.call(this,P,te[0]):(te[1]||kp).call(this,P,te[0]));else throw new Error("Cannot close `"+P.type+"` ("+bi({start:P.start,end:P.end})+"): it’s not open");Y.position.end=sn(P.end)}function d(){return xc(this.stack.pop())}function c(){this.data.expectingFirstListItemValue=!0}function p(P){if(this.data.expectingFirstListItemValue){const M=this.stack[this.stack.length-2];M.start=Number.parseInt(this.sliceSerialize(P),10),this.data.expectingFirstListItemValue=void 0}}function f(){const P=this.resume(),M=this.stack[this.stack.length-1];M.lang=P}function y(){const P=this.resume(),M=this.stack[this.stack.length-1];M.meta=P}function v(){this.data.flowCodeInside||(this.buffer(),this.data.flowCodeInside=!0)}function k(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P.replace(/^(\r?\n|\r)|(\r?\n|\r)$/g,""),this.data.flowCodeInside=void 0}function m(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P.replace(/(\r?\n|\r)$/g,"")}function g(P){const M=this.resume(),Y=this.stack[this.stack.length-1];Y.label=M,Y.identifier=Rt(this.sliceSerialize(P)).toLowerCase()}function x(){const P=this.resume(),M=this.stack[this.stack.length-1];M.title=P}function C(){const P=this.resume(),M=this.stack[this.stack.length-1];M.url=P}function z(P){const M=this.stack[this.stack.length-1];if(!M.depth){const Y=this.sliceSerialize(P).length;M.depth=Y}}function S(){this.data.setextHeadingSlurpLineEnding=!0}function T(P){const M=this.stack[this.stack.length-1];M.depth=this.sliceSerialize(P).codePointAt(0)===61?1:2}function N(){this.data.setextHeadingSlurpLineEnding=void 0}function F(P){const Y=this.stack[this.stack.length-1].children;let te=Y[Y.length-1];(!te||te.type!=="text")&&(te=py(),te.position={start:sn(P.start),end:void 0},Y.push(te)),this.stack.push(te)}function b(P){const M=this.stack.pop();M.value+=this.sliceSerialize(P),M.position.end=sn(P.end)}function I(P){const M=this.stack[this.stack.length-1];if(this.data.atHardBreak){const Y=M.children[M.children.length-1];Y.position.end=sn(P.end),this.data.atHardBreak=void 0;return}!this.data.setextHeadingSlurpLineEnding&&t.canContainEols.includes(M.type)&&(F.call(this,P),b.call(this,P))}function W(){this.data.atHardBreak=!0}function X(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P}function ee(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P}function A(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P}function $(){const P=this.stack[this.stack.length-1];if(this.data.inReference){const M=this.data.referenceType||"shortcut";P.type+="Reference",P.referenceType=M,delete P.url,delete P.title}else delete P.identifier,delete P.label;this.data.referenceType=void 0}function ne(){const P=this.stack[this.stack.length-1];if(this.data.inReference){const M=this.data.referenceType||"shortcut";P.type+="Reference",P.referenceType=M,delete P.url,delete P.title}else delete P.identifier,delete P.label;this.data.referenceType=void 0}function O(P){const M=this.sliceSerialize(P),Y=this.stack[this.stack.length-2];Y.label=Bb(M),Y.identifier=Rt(M).toLowerCase()}function V(){const P=this.stack[this.stack.length-1],M=this.resume(),Y=this.stack[this.stack.length-1];if(this.data.inReference=!0,Y.type==="link"){const te=P.children;Y.children=te}else Y.alt=M}function _(){const P=this.resume(),M=this.stack[this.stack.length-1];M.url=P}function B(){const P=this.resume(),M=this.stack[this.stack.length-1];M.title=P}function Q(){this.data.inReference=void 0}function w(){this.data.referenceType="collapsed"}function ie(P){const M=this.resume(),Y=this.stack[this.stack.length-1];Y.label=M,Y.identifier=Rt(this.sliceSerialize(P)).toLowerCase(),this.data.referenceType="full"}function be(P){this.data.characterReferenceType=P.type}function le(P){const M=this.sliceSerialize(P),Y=this.data.characterReferenceType;let te;Y?(te=rg(M,Y==="characterReferenceMarkerNumeric"?10:16),this.data.characterReferenceType=void 0):te=vc(M);const ue=this.stack[this.stack.length-1];ue.value+=te}function dt(P){const M=this.stack.pop();M.position.end=sn(P.end)}function q(P){b.call(this,P);const M=this.stack[this.stack.length-1];M.url=this.sliceSerialize(P)}function L(P){b.call(this,P);const M=this.stack[this.stack.length-1];M.url="mailto:"+this.sliceSerialize(P)}function D(){return{type:"blockquote",children:[]}}function H(){return{type:"code",lang:null,meta:null,value:""}}function Ee(){return{type:"inlineCode",value:""}}function ve(){return{type:"definition",identifier:"",label:null,title:null,url:""}}function Be(){return{type:"emphasis",children:[]}}function Ot(){return{type:"heading",depth:0,children:[]}}function jn(){return{type:"break"}}function Rn(){return{type:"html",value:""}}function ly(){return{type:"image",title:null,url:"",alt:null}}function Ac(){return{type:"link",title:null,url:"",children:[]}}function Lc(P){return{type:"list",ordered:P.type==="listOrdered",start:null,spread:P._spread,children:[]}}function uy(P){return{type:"listItem",spread:P._spread,checked:null,children:[]}}function cy(){return{type:"paragraph",children:[]}}function dy(){return{type:"strong",children:[]}}function py(){return{type:"text",value:""}}function fy(){return{type:"thematicBreak"}}}function sn(e){return{line:e.line,column:e.column,offset:e.offset}}function mg(e,t){let n=-1;for(;++n<t.length;){const r=t[n];Array.isArray(r)?mg(e,r):$b(e,r)}}function $b(e,t){let n;for(n in t)if(fg.call(t,n))switch(n){case"canContainEols":{const r=t[n];r&&e[n].push(...r);break}case"transforms":{const r=t[n];r&&e[n].push(...r);break}case"enter":case"exit":{const r=t[n];r&&Object.assign(e[n],r);break}}}function kp(e,t){throw e?new Error("Cannot close `"+e.type+"` ("+bi({start:e.start,end:e.end})+"): a different token (`"+t.type+"`, "+bi({start:t.start,end:t.end})+") is open"):new Error("Cannot close document, a token (`"+t.type+"`, "+bi({start:t.start,end:t.end})+") is still open")}function Qb(e){const t=this;t.parser=n;function n(r){return Vb(r,{...t.data("settings"),...e,extensions:t.data("micromarkExtensions")||[],mdastExtensions:t.data("fromMarkdownExtensions")||[]})}}function Kb(e,t){const n={type:"element",tagName:"blockquote",properties:{},children:e.wrap(e.all(t),!0)};return e.patch(t,n),e.applyData(t,n)}function Xb(e,t){const n={type:"element",tagName:"br",properties:{},children:[]};return e.patch(t,n),[e.applyData(t,n),{type:"text",value:`
`}]}function Yb(e,t){const n=t.value?t.value+`
`:"",r={},i=t.lang?t.lang.split(/\s+/):[];i.length>0&&(r.className=["language-"+i[0]]);let a={type:"element",tagName:"code",properties:r,children:[{type:"text",value:n}]};return t.meta&&(a.data={meta:t.meta}),e.patch(t,a),a=e.applyData(t,a),a={type:"element",tagName:"pre",properties:{},children:[a]},e.patch(t,a),a}function Gb(e,t){const n={type:"element",tagName:"del",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function Zb(e,t){const n={type:"element",tagName:"em",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function Jb(e,t){const n=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",r=String(t.identifier).toUpperCase(),i=Hr(r.toLowerCase()),a=e.footnoteOrder.indexOf(r);let o,s=e.footnoteCounts.get(r);s===void 0?(s=0,e.footnoteOrder.push(r),o=e.footnoteOrder.length):o=a+1,s+=1,e.footnoteCounts.set(r,s);const l={type:"element",tagName:"a",properties:{href:"#"+n+"fn-"+i,id:n+"fnref-"+i+(s>1?"-"+s:""),dataFootnoteRef:!0,ariaDescribedBy:["footnote-label"]},children:[{type:"text",value:String(o)}]};e.patch(t,l);const u={type:"element",tagName:"sup",properties:{},children:[l]};return e.patch(t,u),e.applyData(t,u)}function ek(e,t){const n={type:"element",tagName:"h"+t.depth,properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function tk(e,t){if(e.options.allowDangerousHtml){const n={type:"raw",value:t.value};return e.patch(t,n),e.applyData(t,n)}}function hg(e,t){const n=t.referenceType;let r="]";if(n==="collapsed"?r+="[]":n==="full"&&(r+="["+(t.label||t.identifier)+"]"),t.type==="imageReference")return[{type:"text",value:"!["+t.alt+r}];const i=e.all(t),a=i[0];a&&a.type==="text"?a.value="["+a.value:i.unshift({type:"text",value:"["});const o=i[i.length-1];return o&&o.type==="text"?o.value+=r:i.push({type:"text",value:r}),i}function nk(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return hg(e,t);const i={src:Hr(r.url||""),alt:t.alt};r.title!==null&&r.title!==void 0&&(i.title=r.title);const a={type:"element",tagName:"img",properties:i,children:[]};return e.patch(t,a),e.applyData(t,a)}function rk(e,t){const n={src:Hr(t.url)};t.alt!==null&&t.alt!==void 0&&(n.alt=t.alt),t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"img",properties:n,children:[]};return e.patch(t,r),e.applyData(t,r)}function ik(e,t){const n={type:"text",value:t.value.replace(/\r?\n|\r/g," ")};e.patch(t,n);const r={type:"element",tagName:"code",properties:{},children:[n]};return e.patch(t,r),e.applyData(t,r)}function ak(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return hg(e,t);const i={href:Hr(r.url||"")};r.title!==null&&r.title!==void 0&&(i.title=r.title);const a={type:"element",tagName:"a",properties:i,children:e.all(t)};return e.patch(t,a),e.applyData(t,a)}function ok(e,t){const n={href:Hr(t.url)};t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"a",properties:n,children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function sk(e,t,n){const r=e.all(t),i=n?lk(n):gg(t),a={},o=[];if(typeof t.checked=="boolean"){const d=r[0];let c;d&&d.type==="element"&&d.tagName==="p"?c=d:(c={type:"element",tagName:"p",properties:{},children:[]},r.unshift(c)),c.children.length>0&&c.children.unshift({type:"text",value:" "}),c.children.unshift({type:"element",tagName:"input",properties:{type:"checkbox",checked:t.checked,disabled:!0},children:[]}),a.className=["task-list-item"]}let s=-1;for(;++s<r.length;){const d=r[s];(i||s!==0||d.type!=="element"||d.tagName!=="p")&&o.push({type:"text",value:`
`}),d.type==="element"&&d.tagName==="p"&&!i?o.push(...d.children):o.push(d)}const l=r[r.length-1];l&&(i||l.type!=="element"||l.tagName!=="p")&&o.push({type:"text",value:`
`});const u={type:"element",tagName:"li",properties:a,children:o};return e.patch(t,u),e.applyData(t,u)}function lk(e){let t=!1;if(e.type==="list"){t=e.spread||!1;const n=e.children;let r=-1;for(;!t&&++r<n.length;)t=gg(n[r])}return t}function gg(e){const t=e.spread;return t??e.children.length>1}function uk(e,t){const n={},r=e.all(t);let i=-1;for(typeof t.start=="number"&&t.start!==1&&(n.start=t.start);++i<r.length;){const o=r[i];if(o.type==="element"&&o.tagName==="li"&&o.properties&&Array.isArray(o.properties.className)&&o.properties.className.includes("task-list-item")){n.className=["contains-task-list"];break}}const a={type:"element",tagName:t.ordered?"ol":"ul",properties:n,children:e.wrap(r,!0)};return e.patch(t,a),e.applyData(t,a)}function ck(e,t){const n={type:"element",tagName:"p",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function dk(e,t){const n={type:"root",children:e.wrap(e.all(t))};return e.patch(t,n),e.applyData(t,n)}function pk(e,t){const n={type:"element",tagName:"strong",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function fk(e,t){const n=e.all(t),r=n.shift(),i=[];if(r){const o={type:"element",tagName:"thead",properties:{},children:e.wrap([r],!0)};e.patch(t.children[0],o),i.push(o)}if(n.length>0){const o={type:"element",tagName:"tbody",properties:{},children:e.wrap(n,!0)},s=mc(t.children[1]),l=Xh(t.children[t.children.length-1]);s&&l&&(o.position={start:s,end:l}),i.push(o)}const a={type:"element",tagName:"table",properties:{},children:e.wrap(i,!0)};return e.patch(t,a),e.applyData(t,a)}function mk(e,t,n){const r=n?n.children:void 0,a=(r?r.indexOf(t):1)===0?"th":"td",o=n&&n.type==="table"?n.align:void 0,s=o?o.length:t.children.length;let l=-1;const u=[];for(;++l<s;){const c=t.children[l],p={},f=o?o[l]:void 0;f&&(p.align=f);let y={type:"element",tagName:a,properties:p,children:[]};c&&(y.children=e.all(c),e.patch(c,y),y=e.applyData(c,y)),u.push(y)}const d={type:"element",tagName:"tr",properties:{},children:e.wrap(u,!0)};return e.patch(t,d),e.applyData(t,d)}function hk(e,t){const n={type:"element",tagName:"td",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}const Sp=9,Cp=32;function gk(e){const t=String(e),n=/\r?\n|\r/g;let r=n.exec(t),i=0;const a=[];for(;r;)a.push(Ep(t.slice(i,r.index),i>0,!0),r[0]),i=r.index+r[0].length,r=n.exec(t);return a.push(Ep(t.slice(i),i>0,!1)),a.join("")}function Ep(e,t,n){let r=0,i=e.length;if(t){let a=e.codePointAt(r);for(;a===Sp||a===Cp;)r++,a=e.codePointAt(r)}if(n){let a=e.codePointAt(i-1);for(;a===Sp||a===Cp;)i--,a=e.codePointAt(i-1)}return i>r?e.slice(r,i):""}function yk(e,t){const n={type:"text",value:gk(String(t.value))};return e.patch(t,n),e.applyData(t,n)}function xk(e,t){const n={type:"element",tagName:"hr",properties:{},children:[]};return e.patch(t,n),e.applyData(t,n)}const vk={blockquote:Kb,break:Xb,code:Yb,delete:Gb,emphasis:Zb,footnoteReference:Jb,heading:ek,html:tk,imageReference:nk,image:rk,inlineCode:ik,linkReference:ak,link:ok,listItem:sk,list:uk,paragraph:ck,root:dk,strong:pk,table:fk,tableCell:hk,tableRow:mk,text:yk,thematicBreak:xk,toml:Ca,yaml:Ca,definition:Ca,footnoteDefinition:Ca};function Ca(){}const yg=-1,Go=0,Si=1,So=2,wc=3,bc=4,kc=5,Sc=6,xg=7,vg=8,zp=typeof self=="object"?self:globalThis,_k=(e,t)=>{const n=(i,a)=>(e.set(a,i),i),r=i=>{if(e.has(i))return e.get(i);const[a,o]=t[i];switch(a){case Go:case yg:return n(o,i);case Si:{const s=n([],i);for(const l of o)s.push(r(l));return s}case So:{const s=n({},i);for(const[l,u]of o)s[r(l)]=r(u);return s}case wc:return n(new Date(o),i);case bc:{const{source:s,flags:l}=o;return n(new RegExp(s,l),i)}case kc:{const s=n(new Map,i);for(const[l,u]of o)s.set(r(l),r(u));return s}case Sc:{const s=n(new Set,i);for(const l of o)s.add(r(l));return s}case xg:{const{name:s,message:l}=o;return n(new zp[s](l),i)}case vg:return n(BigInt(o),i);case"BigInt":return n(Object(BigInt(o)),i);case"ArrayBuffer":return n(new Uint8Array(o).buffer,o);case"DataView":{const{buffer:s}=new Uint8Array(o);return n(new DataView(s),o)}}return n(new zp[a](o),i)};return r},Tp=e=>_k(new Map,e)(0),ar="",{toString:wk}={},{keys:bk}=Object,ai=e=>{const t=typeof e;if(t!=="object"||!e)return[Go,t];const n=wk.call(e).slice(8,-1);switch(n){case"Array":return[Si,ar];case"Object":return[So,ar];case"Date":return[wc,ar];case"RegExp":return[bc,ar];case"Map":return[kc,ar];case"Set":return[Sc,ar];case"DataView":return[Si,n]}return n.includes("Array")?[Si,n]:n.includes("Error")?[xg,n]:[So,n]},Ea=([e,t])=>e===Go&&(t==="function"||t==="symbol"),kk=(e,t,n,r)=>{const i=(o,s)=>{const l=r.push(o)-1;return n.set(s,l),l},a=o=>{if(n.has(o))return n.get(o);let[s,l]=ai(o);switch(s){case Go:{let d=o;switch(l){case"bigint":s=vg,d=o.toString();break;case"function":case"symbol":if(e)throw new TypeError("unable to serialize "+l);d=null;break;case"undefined":return i([yg],o)}return i([s,d],o)}case Si:{if(l){let p=o;return l==="DataView"?p=new Uint8Array(o.buffer):l==="ArrayBuffer"&&(p=new Uint8Array(o)),i([l,[...p]],o)}const d=[],c=i([s,d],o);for(const p of o)d.push(a(p));return c}case So:{if(l)switch(l){case"BigInt":return i([l,o.toString()],o);case"Boolean":case"Number":case"String":return i([l,o.valueOf()],o)}if(t&&"toJSON"in o)return a(o.toJSON());const d=[],c=i([s,d],o);for(const p of bk(o))(e||!Ea(ai(o[p])))&&d.push([a(p),a(o[p])]);return c}case wc:return i([s,o.toISOString()],o);case bc:{const{source:d,flags:c}=o;return i([s,{source:d,flags:c}],o)}case kc:{const d=[],c=i([s,d],o);for(const[p,f]of o)(e||!(Ea(ai(p))||Ea(ai(f))))&&d.push([a(p),a(f)]);return c}case Sc:{const d=[],c=i([s,d],o);for(const p of o)(e||!Ea(ai(p)))&&d.push(a(p));return c}}const{message:u}=o;return i([s,{name:l,message:u}],o)};return a},Pp=(e,{json:t,lossy:n}={})=>{const r=[];return kk(!(t||n),!!t,new Map,r)(e),r},Co=typeof structuredClone=="function"?(e,t)=>t&&("json"in t||"lossy"in t)?Tp(Pp(e,t)):structuredClone(e):(e,t)=>Tp(Pp(e,t));function Sk(e,t){const n=[{type:"text",value:"↩"}];return t>1&&n.push({type:"element",tagName:"sup",properties:{},children:[{type:"text",value:String(t)}]}),n}function Ck(e,t){return"Back to reference "+(e+1)+(t>1?"-"+t:"")}function Ek(e){const t=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",n=e.options.footnoteBackContent||Sk,r=e.options.footnoteBackLabel||Ck,i=e.options.footnoteLabel||"Footnotes",a=e.options.footnoteLabelTagName||"h2",o=e.options.footnoteLabelProperties||{className:["sr-only"]},s=[];let l=-1;for(;++l<e.footnoteOrder.length;){const u=e.footnoteById.get(e.footnoteOrder[l]);if(!u)continue;const d=e.all(u),c=String(u.identifier).toUpperCase(),p=Hr(c.toLowerCase());let f=0;const y=[],v=e.footnoteCounts.get(c);for(;v!==void 0&&++f<=v;){y.length>0&&y.push({type:"text",value:" "});let g=typeof n=="string"?n:n(l,f);typeof g=="string"&&(g={type:"text",value:g}),y.push({type:"element",tagName:"a",properties:{href:"#"+t+"fnref-"+p+(f>1?"-"+f:""),dataFootnoteBackref:"",ariaLabel:typeof r=="string"?r:r(l,f),className:["data-footnote-backref"]},children:Array.isArray(g)?g:[g]})}const k=d[d.length-1];if(k&&k.type==="element"&&k.tagName==="p"){const g=k.children[k.children.length-1];g&&g.type==="text"?g.value+=" ":k.children.push({type:"text",value:" "}),k.children.push(...y)}else d.push(...y);const m={type:"element",tagName:"li",properties:{id:t+"fn-"+p},children:e.wrap(d,!0)};e.patch(u,m),s.push(m)}if(s.length!==0)return{type:"element",tagName:"section",properties:{dataFootnotes:!0,className:["footnotes"]},children:[{type:"element",tagName:a,properties:{...Co(o),id:"footnote-label"},children:[{type:"text",value:i}]},{type:"text",value:`
`},{type:"element",tagName:"ol",properties:{},children:e.wrap(s,!0)},{type:"text",value:`
`}]}}const Zo=function(e){if(e==null)return Nk;if(typeof e=="function")return Jo(e);if(typeof e=="object")return Array.isArray(e)?zk(e):Tk(e);if(typeof e=="string")return Pk(e);throw new Error("Expected function, string, or object as test")};function zk(e){const t=[];let n=-1;for(;++n<e.length;)t[n]=Zo(e[n]);return Jo(r);function r(...i){let a=-1;for(;++a<t.length;)if(t[a].apply(this,i))return!0;return!1}}function Tk(e){const t=e;return Jo(n);function n(r){const i=r;let a;for(a in e)if(i[a]!==t[a])return!1;return!0}}function Pk(e){return Jo(t);function t(n){return n&&n.type===e}}function Jo(e){return t;function t(n,r,i){return!!(Ak(n)&&e.call(this,n,typeof r=="number"?r:void 0,i||void 0))}}function Nk(){return!0}function Ak(e){return e!==null&&typeof e=="object"&&"type"in e}const _g=[],Lk=!0,lu=!1,jk="skip";function wg(e,t,n,r){let i;typeof t=="function"&&typeof n!="function"?(r=n,n=t):i=t;const a=Zo(i),o=r?-1:1;s(e,void 0,[])();function s(l,u,d){const c=l&&typeof l=="object"?l:{};if(typeof c.type=="string"){const f=typeof c.tagName=="string"?c.tagName:typeof c.name=="string"?c.name:void 0;Object.defineProperty(p,"name",{value:"node ("+(l.type+(f?"<"+f+">":""))+")"})}return p;function p(){let f=_g,y,v,k;if((!t||a(l,u,d[d.length-1]||void 0))&&(f=Rk(n(l,d)),f[0]===lu))return f;if("children"in l&&l.children){const m=l;if(m.children&&f[0]!==jk)for(v=(r?m.children.length:-1)+o,k=d.concat(m);v>-1&&v<m.children.length;){const g=m.children[v];if(y=s(g,v,k)(),y[0]===lu)return y;v=typeof y[1]=="number"?y[1]:v+o}}return f}}}function Rk(e){return Array.isArray(e)?e:typeof e=="number"?[Lk,e]:e==null?_g:[e]}function Cc(e,t,n,r){let i,a,o;typeof t=="function"&&typeof n!="function"?(a=void 0,o=t,i=n):(a=t,o=n,i=r),wg(e,a,s,i);function s(l,u){const d=u[u.length-1],c=d?d.children.indexOf(l):void 0;return o(l,c,d)}}const uu={}.hasOwnProperty,Ik={};function Ok(e,t){const n=t||Ik,r=new Map,i=new Map,a=new Map,o={...vk,...n.handlers},s={all:u,applyData:Mk,definitionById:r,footnoteById:i,footnoteCounts:a,footnoteOrder:[],handlers:o,one:l,options:n,patch:Dk,wrap:qk};return Cc(e,function(d){if(d.type==="definition"||d.type==="footnoteDefinition"){const c=d.type==="definition"?r:i,p=String(d.identifier).toUpperCase();c.has(p)||c.set(p,d)}}),s;function l(d,c){const p=d.type,f=s.handlers[p];if(uu.call(s.handlers,p)&&f)return f(s,d,c);if(s.options.passThrough&&s.options.passThrough.includes(p)){if("children"in d){const{children:v,...k}=d,m=Co(k);return m.children=s.all(d),m}return Co(d)}return(s.options.unknownHandler||Fk)(s,d,c)}function u(d){const c=[];if("children"in d){const p=d.children;let f=-1;for(;++f<p.length;){const y=s.one(p[f],d);if(y){if(f&&p[f-1].type==="break"&&(!Array.isArray(y)&&y.type==="text"&&(y.value=Np(y.value)),!Array.isArray(y)&&y.type==="element")){const v=y.children[0];v&&v.type==="text"&&(v.value=Np(v.value))}Array.isArray(y)?c.push(...y):c.push(y)}}}return c}}function Dk(e,t){e.position&&(t.position=C_(e))}function Mk(e,t){let n=t;if(e&&e.data){const r=e.data.hName,i=e.data.hChildren,a=e.data.hProperties;if(typeof r=="string")if(n.type==="element")n.tagName=r;else{const o="children"in n?n.children:[n];n={type:"element",tagName:r,properties:{},children:o}}n.type==="element"&&a&&Object.assign(n.properties,Co(a)),"children"in n&&n.children&&i!==null&&i!==void 0&&(n.children=i)}return n}function Fk(e,t){const n=t.data||{},r="value"in t&&!(uu.call(n,"hProperties")||uu.call(n,"hChildren"))?{type:"text",value:t.value}:{type:"element",tagName:"div",properties:{},children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function qk(e,t){const n=[];let r=-1;for(t&&n.push({type:"text",value:`
`});++r<e.length;)r&&n.push({type:"text",value:`
`}),n.push(e[r]);return t&&e.length>0&&n.push({type:"text",value:`
`}),n}function Np(e){let t=0,n=e.charCodeAt(t);for(;n===9||n===32;)t++,n=e.charCodeAt(t);return e.slice(t)}function Ap(e,t){const n=Ok(e,t),r=n.one(e,void 0),i=Ek(n),a=Array.isArray(r)?{type:"root",children:r}:r||{type:"root",children:[]};return i&&a.children.push({type:"text",value:`
`},i),a}function Wk(e,t){return e&&"run"in e?async function(n,r){const i=Ap(n,{file:r,...t});await e.run(i,r)}:function(n,r){return Ap(n,{file:r,...e||t})}}function Lp(e){if(e)throw e}var Ka=Object.prototype.hasOwnProperty,bg=Object.prototype.toString,jp=Object.defineProperty,Rp=Object.getOwnPropertyDescriptor,Ip=function(t){return typeof Array.isArray=="function"?Array.isArray(t):bg.call(t)==="[object Array]"},Op=function(t){if(!t||bg.call(t)!=="[object Object]")return!1;var n=Ka.call(t,"constructor"),r=t.constructor&&t.constructor.prototype&&Ka.call(t.constructor.prototype,"isPrototypeOf");if(t.constructor&&!n&&!r)return!1;var i;for(i in t);return typeof i>"u"||Ka.call(t,i)},Dp=function(t,n){jp&&n.name==="__proto__"?jp(t,n.name,{enumerable:!0,configurable:!0,value:n.newValue,writable:!0}):t[n.name]=n.newValue},Mp=function(t,n){if(n==="__proto__")if(Ka.call(t,n)){if(Rp)return Rp(t,n).value}else return;return t[n]},Bk=function e(){var t,n,r,i,a,o,s=arguments[0],l=1,u=arguments.length,d=!1;for(typeof s=="boolean"&&(d=s,s=arguments[1]||{},l=2),(s==null||typeof s!="object"&&typeof s!="function")&&(s={});l<u;++l)if(t=arguments[l],t!=null)for(n in t)r=Mp(s,n),i=Mp(t,n),s!==i&&(d&&i&&(Op(i)||(a=Ip(i)))?(a?(a=!1,o=r&&Ip(r)?r:[]):o=r&&Op(r)?r:{},Dp(s,{name:n,newValue:e(d,o,i)})):typeof i<"u"&&Dp(s,{name:n,newValue:i}));return s};const Ds=Zn(Bk);function cu(e){if(typeof e!="object"||e===null)return!1;const t=Object.getPrototypeOf(e);return(t===null||t===Object.prototype||Object.getPrototypeOf(t)===null)&&!(Symbol.toStringTag in e)&&!(Symbol.iterator in e)}function Uk(){const e=[],t={run:n,use:r};return t;function n(...i){let a=-1;const o=i.pop();if(typeof o!="function")throw new TypeError("Expected function as last argument, not "+o);s(null,...i);function s(l,...u){const d=e[++a];let c=-1;if(l){o(l);return}for(;++c<i.length;)(u[c]===null||u[c]===void 0)&&(u[c]=i[c]);i=u,d?Vk(d,s)(...u):o(null,...u)}}function r(i){if(typeof i!="function")throw new TypeError("Expected `middelware` to be a function, not "+i);return e.push(i),t}}function Vk(e,t){let n;return r;function r(...o){const s=e.length>o.length;let l;s&&o.push(i);try{l=e.apply(this,o)}catch(u){const d=u;if(s&&n)throw d;return i(d)}s||(l&&l.then&&typeof l.then=="function"?l.then(a,i):l instanceof Error?i(l):a(l))}function i(o,...s){n||(n=!0,t(o,...s))}function a(o){i(null,o)}}const Ft={basename:Hk,dirname:$k,extname:Qk,join:Kk,sep:"/"};function Hk(e,t){if(t!==void 0&&typeof t!="string")throw new TypeError('"ext" argument must be a string');aa(e);let n=0,r=-1,i=e.length,a;if(t===void 0||t.length===0||t.length>e.length){for(;i--;)if(e.codePointAt(i)===47){if(a){n=i+1;break}}else r<0&&(a=!0,r=i+1);return r<0?"":e.slice(n,r)}if(t===e)return"";let o=-1,s=t.length-1;for(;i--;)if(e.codePointAt(i)===47){if(a){n=i+1;break}}else o<0&&(a=!0,o=i+1),s>-1&&(e.codePointAt(i)===t.codePointAt(s--)?s<0&&(r=i):(s=-1,r=o));return n===r?r=o:r<0&&(r=e.length),e.slice(n,r)}function $k(e){if(aa(e),e.length===0)return".";let t=-1,n=e.length,r;for(;--n;)if(e.codePointAt(n)===47){if(r){t=n;break}}else r||(r=!0);return t<0?e.codePointAt(0)===47?"/":".":t===1&&e.codePointAt(0)===47?"//":e.slice(0,t)}function Qk(e){aa(e);let t=e.length,n=-1,r=0,i=-1,a=0,o;for(;t--;){const s=e.codePointAt(t);if(s===47){if(o){r=t+1;break}continue}n<0&&(o=!0,n=t+1),s===46?i<0?i=t:a!==1&&(a=1):i>-1&&(a=-1)}return i<0||n<0||a===0||a===1&&i===n-1&&i===r+1?"":e.slice(i,n)}function Kk(...e){let t=-1,n;for(;++t<e.length;)aa(e[t]),e[t]&&(n=n===void 0?e[t]:n+"/"+e[t]);return n===void 0?".":Xk(n)}function Xk(e){aa(e);const t=e.codePointAt(0)===47;let n=Yk(e,!t);return n.length===0&&!t&&(n="."),n.length>0&&e.codePointAt(e.length-1)===47&&(n+="/"),t?"/"+n:n}function Yk(e,t){let n="",r=0,i=-1,a=0,o=-1,s,l;for(;++o<=e.length;){if(o<e.length)s=e.codePointAt(o);else{if(s===47)break;s=47}if(s===47){if(!(i===o-1||a===1))if(i!==o-1&&a===2){if(n.length<2||r!==2||n.codePointAt(n.length-1)!==46||n.codePointAt(n.length-2)!==46){if(n.length>2){if(l=n.lastIndexOf("/"),l!==n.length-1){l<0?(n="",r=0):(n=n.slice(0,l),r=n.length-1-n.lastIndexOf("/")),i=o,a=0;continue}}else if(n.length>0){n="",r=0,i=o,a=0;continue}}t&&(n=n.length>0?n+"/..":"..",r=2)}else n.length>0?n+="/"+e.slice(i+1,o):n=e.slice(i+1,o),r=o-i-1;i=o,a=0}else s===46&&a>-1?a++:a=-1}return n}function aa(e){if(typeof e!="string")throw new TypeError("Path must be a string. Received "+JSON.stringify(e))}const Gk={cwd:Zk};function Zk(){return"/"}function du(e){return!!(e!==null&&typeof e=="object"&&"href"in e&&e.href&&"protocol"in e&&e.protocol&&e.auth===void 0)}function Jk(e){if(typeof e=="string")e=new URL(e);else if(!du(e)){const t=new TypeError('The "path" argument must be of type string or an instance of URL. Received `'+e+"`");throw t.code="ERR_INVALID_ARG_TYPE",t}if(e.protocol!=="file:"){const t=new TypeError("The URL must be of scheme file");throw t.code="ERR_INVALID_URL_SCHEME",t}return e2(e)}function e2(e){if(e.hostname!==""){const r=new TypeError('File URL host must be "localhost" or empty on darwin');throw r.code="ERR_INVALID_FILE_URL_HOST",r}const t=e.pathname;let n=-1;for(;++n<t.length;)if(t.codePointAt(n)===37&&t.codePointAt(n+1)===50){const r=t.codePointAt(n+2);if(r===70||r===102){const i=new TypeError("File URL path must not include encoded / characters");throw i.code="ERR_INVALID_FILE_URL_PATH",i}}return decodeURIComponent(t)}const Ms=["history","path","basename","stem","extname","dirname"];class kg{constructor(t){let n;t?du(t)?n={path:t}:typeof t=="string"||t2(t)?n={value:t}:n=t:n={},this.cwd="cwd"in n?"":Gk.cwd(),this.data={},this.history=[],this.messages=[],this.value,this.map,this.result,this.stored;let r=-1;for(;++r<Ms.length;){const a=Ms[r];a in n&&n[a]!==void 0&&n[a]!==null&&(this[a]=a==="history"?[...n[a]]:n[a])}let i;for(i in n)Ms.includes(i)||(this[i]=n[i])}get basename(){return typeof this.path=="string"?Ft.basename(this.path):void 0}set basename(t){qs(t,"basename"),Fs(t,"basename"),this.path=Ft.join(this.dirname||"",t)}get dirname(){return typeof this.path=="string"?Ft.dirname(this.path):void 0}set dirname(t){Fp(this.basename,"dirname"),this.path=Ft.join(t||"",this.basename)}get extname(){return typeof this.path=="string"?Ft.extname(this.path):void 0}set extname(t){if(Fs(t,"extname"),Fp(this.dirname,"extname"),t){if(t.codePointAt(0)!==46)throw new Error("`extname` must start with `.`");if(t.includes(".",1))throw new Error("`extname` cannot contain multiple dots")}this.path=Ft.join(this.dirname,this.stem+(t||""))}get path(){return this.history[this.history.length-1]}set path(t){du(t)&&(t=Jk(t)),qs(t,"path"),this.path!==t&&this.history.push(t)}get stem(){return typeof this.path=="string"?Ft.basename(this.path,this.extname):void 0}set stem(t){qs(t,"stem"),Fs(t,"stem"),this.path=Ft.join(this.dirname||"",t+(this.extname||""))}fail(t,n,r){const i=this.message(t,n,r);throw i.fatal=!0,i}info(t,n,r){const i=this.message(t,n,r);return i.fatal=void 0,i}message(t,n,r){const i=new We(t,n,r);return this.path&&(i.name=this.path+":"+i.name,i.file=this.path),i.fatal=!1,this.messages.push(i),i}toString(t){return this.value===void 0?"":typeof this.value=="string"?this.value:new TextDecoder(t||void 0).decode(this.value)}}function Fs(e,t){if(e&&e.includes(Ft.sep))throw new Error("`"+t+"` cannot be a path: did not expect `"+Ft.sep+"`")}function qs(e,t){if(!e)throw new Error("`"+t+"` cannot be empty")}function Fp(e,t){if(!e)throw new Error("Setting `"+t+"` requires `path` to be set too")}function t2(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const n2=function(e){const r=this.constructor.prototype,i=r[e],a=function(){return i.apply(a,arguments)};return Object.setPrototypeOf(a,r),a},r2={}.hasOwnProperty;class Ec extends n2{constructor(){super("copy"),this.Compiler=void 0,this.Parser=void 0,this.attachers=[],this.compiler=void 0,this.freezeIndex=-1,this.frozen=void 0,this.namespace={},this.parser=void 0,this.transformers=Uk()}copy(){const t=new Ec;let n=-1;for(;++n<this.attachers.length;){const r=this.attachers[n];t.use(...r)}return t.data(Ds(!0,{},this.namespace)),t}data(t,n){return typeof t=="string"?arguments.length===2?(Us("data",this.frozen),this.namespace[t]=n,this):r2.call(this.namespace,t)&&this.namespace[t]||void 0:t?(Us("data",this.frozen),this.namespace=t,this):this.namespace}freeze(){if(this.frozen)return this;const t=this;for(;++this.freezeIndex<this.attachers.length;){const[n,...r]=this.attachers[this.freezeIndex];if(r[0]===!1)continue;r[0]===!0&&(r[0]=void 0);const i=n.call(t,...r);typeof i=="function"&&this.transformers.use(i)}return this.frozen=!0,this.freezeIndex=Number.POSITIVE_INFINITY,this}parse(t){this.freeze();const n=za(t),r=this.parser||this.Parser;return Ws("parse",r),r(String(n),n)}process(t,n){const r=this;return this.freeze(),Ws("process",this.parser||this.Parser),Bs("process",this.compiler||this.Compiler),n?i(void 0,n):new Promise(i);function i(a,o){const s=za(t),l=r.parse(s);r.run(l,s,function(d,c,p){if(d||!c||!p)return u(d);const f=c,y=r.stringify(f,p);o2(y)?p.value=y:p.result=y,u(d,p)});function u(d,c){d||!c?o(d):a?a(c):n(void 0,c)}}}processSync(t){let n=!1,r;return this.freeze(),Ws("processSync",this.parser||this.Parser),Bs("processSync",this.compiler||this.Compiler),this.process(t,i),Wp("processSync","process",n),r;function i(a,o){n=!0,Lp(a),r=o}}run(t,n,r){qp(t),this.freeze();const i=this.transformers;return!r&&typeof n=="function"&&(r=n,n=void 0),r?a(void 0,r):new Promise(a);function a(o,s){const l=za(n);i.run(t,l,u);function u(d,c,p){const f=c||t;d?s(d):o?o(f):r(void 0,f,p)}}}runSync(t,n){let r=!1,i;return this.run(t,n,a),Wp("runSync","run",r),i;function a(o,s){Lp(o),i=s,r=!0}}stringify(t,n){this.freeze();const r=za(n),i=this.compiler||this.Compiler;return Bs("stringify",i),qp(t),i(t,r)}use(t,...n){const r=this.attachers,i=this.namespace;if(Us("use",this.frozen),t!=null)if(typeof t=="function")l(t,n);else if(typeof t=="object")Array.isArray(t)?s(t):o(t);else throw new TypeError("Expected usable value, not `"+t+"`");return this;function a(u){if(typeof u=="function")l(u,[]);else if(typeof u=="object")if(Array.isArray(u)){const[d,...c]=u;l(d,c)}else o(u);else throw new TypeError("Expected usable value, not `"+u+"`")}function o(u){if(!("plugins"in u)&&!("settings"in u))throw new Error("Expected usable value but received an empty preset, which is probably a mistake: presets typically come with `plugins` and sometimes with `settings`, but this has neither");s(u.plugins),u.settings&&(i.settings=Ds(!0,i.settings,u.settings))}function s(u){let d=-1;if(u!=null)if(Array.isArray(u))for(;++d<u.length;){const c=u[d];a(c)}else throw new TypeError("Expected a list of plugins, not `"+u+"`")}function l(u,d){let c=-1,p=-1;for(;++c<r.length;)if(r[c][0]===u){p=c;break}if(p===-1)r.push([u,...d]);else if(d.length>0){let[f,...y]=d;const v=r[p][1];cu(v)&&cu(f)&&(f=Ds(!0,v,f)),r[p]=[u,f,...y]}}}}const i2=new Ec().freeze();function Ws(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `parser`")}function Bs(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `compiler`")}function Us(e,t){if(t)throw new Error("Cannot call `"+e+"` on a frozen processor.\nCreate a new processor first, by calling it: use `processor()` instead of `processor`.")}function qp(e){if(!cu(e)||typeof e.type!="string")throw new TypeError("Expected node, got `"+e+"`")}function Wp(e,t,n){if(!n)throw new Error("`"+e+"` finished async. Use `"+t+"` instead")}function za(e){return a2(e)?e:new kg(e)}function a2(e){return!!(e&&typeof e=="object"&&"message"in e&&"messages"in e)}function o2(e){return typeof e=="string"||s2(e)}function s2(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const l2="https://github.com/remarkjs/react-markdown/blob/main/changelog.md",Bp=[],Up={allowDangerousHtml:!0},u2=/^(https?|ircs?|mailto|xmpp)$/i,c2=[{from:"astPlugins",id:"remove-buggy-html-in-markdown-parser"},{from:"allowDangerousHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"allowNode",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowElement"},{from:"allowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowedElements"},{from:"disallowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"disallowedElements"},{from:"escapeHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"includeElementIndex",id:"#remove-includeelementindex"},{from:"includeNodeIndex",id:"change-includenodeindex-to-includeelementindex"},{from:"linkTarget",id:"remove-linktarget"},{from:"plugins",id:"change-plugins-to-remarkplugins",to:"remarkPlugins"},{from:"rawSourcePos",id:"#remove-rawsourcepos"},{from:"renderers",id:"change-renderers-to-components",to:"components"},{from:"source",id:"change-source-to-children",to:"children"},{from:"sourcePos",id:"#remove-sourcepos"},{from:"transformImageUri",id:"#add-urltransform",to:"urlTransform"},{from:"transformLinkUri",id:"#add-urltransform",to:"urlTransform"}];function Sg(e){const t=d2(e),n=p2(e);return f2(t.runSync(t.parse(n),n),e)}function d2(e){const t=e.rehypePlugins||Bp,n=e.remarkPlugins||Bp,r=e.remarkRehypeOptions?{...e.remarkRehypeOptions,...Up}:Up;return i2().use(Qb).use(n).use(Wk,r).use(t)}function p2(e){const t=e.children||"",n=new kg;return typeof t=="string"&&(n.value=t),n}function f2(e,t){const n=t.allowedElements,r=t.allowElement,i=t.components,a=t.disallowedElements,o=t.skipHtml,s=t.unwrapDisallowed,l=t.urlTransform||m2;for(const d of c2)Object.hasOwn(t,d.from)&&(""+d.from+(d.to?"use `"+d.to+"` instead":"remove it")+l2+d.id,void 0);return t.className&&(e={type:"element",tagName:"div",properties:{className:t.className},children:e.type==="root"?e.children:[e]}),Cc(e,u),N_(e,{Fragment:h.Fragment,components:i,ignoreInvalidStyle:!0,jsx:h.jsx,jsxs:h.jsxs,passKeys:!0,passNode:!0});function u(d,c,p){if(d.type==="raw"&&p&&typeof c=="number")return o?p.children.splice(c,1):p.children[c]={type:"text",value:d.value},c;if(d.type==="element"){let f;for(f in Rs)if(Object.hasOwn(Rs,f)&&Object.hasOwn(d.properties,f)){const y=d.properties[f],v=Rs[f];(v===null||v.includes(d.tagName))&&(d.properties[f]=l(String(y||""),f,d))}}if(d.type==="element"){let f=n?!n.includes(d.tagName):a?a.includes(d.tagName):!1;if(!f&&r&&typeof c=="number"&&(f=!r(d,c,p)),f&&p&&typeof c=="number")return s&&d.children?p.children.splice(c,1,...d.children):p.children.splice(c,1),c}}}function m2(e){const t=e.indexOf(":"),n=e.indexOf("?"),r=e.indexOf("#"),i=e.indexOf("/");return t===-1||i!==-1&&t>i||n!==-1&&t>n||r!==-1&&t>r||u2.test(e.slice(0,t))?e:""}const h2=[{id:"numpy-array-sum",title:"Array Sum",section:"python-basics",difficulty:"easy",description:`
## Array Sum

Given a NumPy array of numbers, implement a function that returns the sum of all elements.

### Function Signature
\`\`\`python
def array_sum(arr: np.ndarray) -> float:
\`\`\`

### Constraints
- Array length: 1 <= n <= 1000
- Element values: -10^6 <= arr[i] <= 10^6
    `,examples:[{input:"np.array([1, 2, 3, 4, 5])",output:"15",explanation:"1 + 2 + 3 + 4 + 5 = 15"},{input:"np.array([-1, 0, 1])",output:"0",explanation:"-1 + 0 + 1 = 0"}],starterCode:`import numpy as np

def array_sum(arr: np.ndarray) -> float:
    """
    Calculate the sum of all elements in a NumPy array.

    Args:
        arr: A NumPy array of numbers

    Returns:
        The sum of all elements
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic positive numbers",input:"[1, 2, 3, 4, 5]",expected:"15",hidden:!1},{id:"2",description:"With negatives",input:"[-1, 0, 1]",expected:"0",hidden:!1},{id:"3",description:"Single element",input:"[42]",expected:"42",hidden:!1},{id:"4",description:"Larger array",input:"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",expected:"55",hidden:!0}],hints:["NumPy arrays have built-in methods for common operations.","Try using np.sum() or the .sum() method on the array."],solution:`import numpy as np

def array_sum(arr: np.ndarray) -> float:
    """
    Calculate the sum of all elements in a NumPy array.
    """
    return np.sum(arr)
`},{id:"numpy-matrix-multiply",title:"Matrix Multiplication",section:"python-basics",difficulty:"medium",description:`
## Matrix Multiplication

Implement matrix multiplication between two 2D NumPy arrays.

### Function Signature
\`\`\`python
def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- A has shape (m, n)
- B has shape (n, p)
- Result has shape (m, p)
- 1 <= m, n, p <= 100

### Note
You should use NumPy's built-in functions, not manual loops.
    `,examples:[{input:"A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]",output:"[[19, 22], [43, 50]]",explanation:"Standard matrix multiplication"}],starterCode:`import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A and B.

    Args:
        A: First matrix of shape (m, n)
        B: Second matrix of shape (n, p)

    Returns:
        Result matrix of shape (m, p)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"2x2 matrices",input:"([[1, 2], [3, 4]], [[5, 6], [7, 8]])",expected:"[[19, 22], [43, 50]]",hidden:!1},{id:"2",description:"Identity multiplication",input:"([[1, 0], [0, 1]], [[5, 6], [7, 8]])",expected:"[[5, 6], [7, 8]]",hidden:!1},{id:"3",description:"Different dimensions",input:"([[1, 2, 3]], [[1], [2], [3]])",expected:"[[14]]",hidden:!0}],hints:["NumPy provides several ways to do matrix multiplication.","Try np.dot(), np.matmul(), or the @ operator."],solution:`import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A and B.
    """
    return np.dot(A, B)
    # Alternative: return A @ B
    # Alternative: return np.matmul(A, B)
`},{id:"numpy-broadcast-add",title:"Broadcasting Addition",section:"python-basics",difficulty:"medium",description:`
## Broadcasting Addition

Given a 2D matrix and a 1D vector, add the vector to each row of the matrix using broadcasting.

### Function Signature
\`\`\`python
def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Matrix has shape (m, n)
- Vector has shape (n,)
- 1 <= m, n <= 100

### Note
Broadcasting is a powerful NumPy feature that allows operations between arrays of different shapes.
    `,examples:[{input:"matrix = [[1, 2, 3], [4, 5, 6]], vector = [10, 20, 30]",output:"[[11, 22, 33], [14, 25, 36]]",explanation:"Vector is added to each row"}],starterCode:`import numpy as np

def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a vector to each row of a matrix using broadcasting.

    Args:
        matrix: 2D array of shape (m, n)
        vector: 1D array of shape (n,)

    Returns:
        Result matrix of shape (m, n)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic broadcasting",input:"([[1, 2, 3], [4, 5, 6]], [10, 20, 30])",expected:"[[11, 22, 33], [14, 25, 36]]",hidden:!1},{id:"2",description:"With zeros",input:"([[1, 2], [3, 4]], [0, 0])",expected:"[[1, 2], [3, 4]]",hidden:!1},{id:"3",description:"Negative values",input:"([[1, 2], [3, 4]], [-1, -2])",expected:"[[0, 0], [2, 2]]",hidden:!0}],hints:["NumPy automatically broadcasts the vector across rows.","Simply use the + operator - NumPy handles the rest!"],solution:`import numpy as np

def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a vector to each row of a matrix using broadcasting.
    """
    return matrix + vector
`}],g2=[{id:"normalize-features",title:"Normalize Features",section:"data-preprocessing",difficulty:"easy",description:`
## Normalize Features (Min-Max Scaling)

Implement min-max normalization to scale features to the range [0, 1].

### Formula
\`\`\`
X_normalized = (X - X_min) / (X_max - X_min)
\`\`\`

### Function Signature
\`\`\`python
def normalize(arr: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Input is a 1D array
- Array length: 2 <= n <= 1000
- All elements are finite numbers
    `,examples:[{input:"np.array([1, 2, 3, 4, 5])",output:"[0.0, 0.25, 0.5, 0.75, 1.0]",explanation:"min=1, max=5, so (x-1)/(5-1) gives [0, 0.25, 0.5, 0.75, 1]"}],starterCode:`import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scale values to [0, 1].

    Args:
        arr: Input array of numbers

    Returns:
        Normalized array with values in [0, 1]
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic normalization",input:"[1, 2, 3, 4, 5]",expected:"[0.0, 0.25, 0.5, 0.75, 1.0]",hidden:!1},{id:"2",description:"With negative values",input:"[-10, 0, 10]",expected:"[0.0, 0.5, 1.0]",hidden:!1},{id:"3",description:"Larger range",input:"[0, 50, 100]",expected:"[0.0, 0.5, 1.0]",hidden:!0}],hints:["Use np.min() and np.max() to find the range.","Apply the formula: (x - min) / (max - min)"],solution:`import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scale values to [0, 1].
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)
`},{id:"handle-missing-data",title:"Handle Missing Data",section:"data-preprocessing",difficulty:"medium",description:`
## Handle Missing Data

Replace NaN (missing) values in an array with the mean of non-NaN values.

### Function Signature
\`\`\`python
def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Input is a 1D array
- Array contains at least one non-NaN value
- NaN values are represented as np.nan
    `,examples:[{input:"np.array([1.0, np.nan, 3.0, np.nan, 5.0])",output:"[1.0, 3.0, 3.0, 3.0, 5.0]",explanation:"Mean of [1, 3, 5] is 3, which replaces NaN values"}],starterCode:`import numpy as np

def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN values with the mean of non-NaN values.

    Args:
        arr: Input array that may contain NaN values

    Returns:
        Array with NaN values replaced by the mean
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic case",input:'[1.0, float("nan"), 3.0, float("nan"), 5.0]',expected:"[1.0, 3.0, 3.0, 3.0, 5.0]",hidden:!1},{id:"2",description:"Single NaN",input:'[2.0, 4.0, float("nan"), 6.0]',expected:"[2.0, 4.0, 4.0, 6.0]",hidden:!1},{id:"3",description:"NaN at start",input:'[float("nan"), 10.0, 20.0]',expected:"[15.0, 10.0, 20.0]",hidden:!0}],hints:["Use np.isnan() to find NaN values.","Use np.nanmean() to compute mean ignoring NaN values.","Use boolean indexing to replace values."],solution:`import numpy as np

def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN values with the mean of non-NaN values.
    """
    arr = arr.copy()  # Don't modify original
    mean_val = np.nanmean(arr)
    arr[np.isnan(arr)] = mean_val
    return arr
`},{id:"one-hot-encode",title:"One-Hot Encoding",section:"data-preprocessing",difficulty:"medium",description:`
## One-Hot Encoding

Convert an array of categorical integers to one-hot encoded format.

### Function Signature
\`\`\`python
def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
\`\`\`

### Example
For labels [0, 1, 2] with 3 classes:
\`\`\`
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]
\`\`\`

### Constraints
- Labels are integers in range [0, num_classes - 1]
- 1 <= num_classes <= 100
    `,examples:[{input:"labels = [0, 1, 2, 1], num_classes = 3",output:"[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]",explanation:"Each label becomes a row with 1 at the label index"}],starterCode:`import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.

    Args:
        labels: 1D array of integer labels
        num_classes: Total number of classes

    Returns:
        2D array of shape (len(labels), num_classes)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic encoding",input:"([0, 1, 2], 3)",expected:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",hidden:!1},{id:"2",description:"Repeated labels",input:"([0, 0, 1, 1], 2)",expected:"[[1, 0], [1, 0], [0, 1], [0, 1]]",hidden:!1},{id:"3",description:"More classes than used",input:"([0, 2], 4)",expected:"[[1, 0, 0, 0], [0, 0, 1, 0]]",hidden:!0}],hints:["Create a zeros matrix of shape (num_samples, num_classes).","Use array indexing to set the appropriate positions to 1.","np.eye() can be useful for creating one-hot encodings."],solution:`import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.
    """
    # Method 1: Using np.eye
    return np.eye(num_classes)[labels].astype(int)

    # Method 2: Manual approach
    # one_hot = np.zeros((len(labels), num_classes), dtype=int)
    # one_hot[np.arange(len(labels)), labels] = 1
    # return one_hot
`}],y2=[{id:"linear-regression-gd",title:"Linear Regression with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
## Linear Regression with Gradient Descent

Implement simple linear regression using gradient descent optimization.

### Model
\`\`\`
y = w * x + b
\`\`\`

### Gradient Descent Updates
\`\`\`
w = w - learning_rate * dw
b = b - learning_rate * db
\`\`\`

Where:
- dw = (2/n) * sum((y_pred - y) * x)
- db = (2/n) * sum(y_pred - y)

### Function Signature
\`\`\`python
def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float, iterations: int) -> tuple:
\`\`\`

Returns (w, b) - the learned weight and bias.
    `,examples:[{input:"X = [1, 2, 3, 4], y = [2, 4, 6, 8], lr = 0.1, iters = 1000",output:"w ≈ 2.0, b ≈ 0.0",explanation:"The true relationship is y = 2x, so w should be close to 2"}],starterCode:`import numpy as np

def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.

    Args:
        X: Input features (1D array)
        y: Target values (1D array)
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations

    Returns:
        Tuple of (weight, bias)
    """
    # Initialize parameters
    w = 0.0
    b = 0.0
    n = len(X)

    # Your gradient descent implementation here
    pass

    return (round(w, 2), round(b, 2))
`,testCases:[{id:"1",description:"Perfect linear relationship",input:"([1, 2, 3, 4], [2, 4, 6, 8], 0.1, 1000)",expected:"(2.0, 0.0)",hidden:!1},{id:"2",description:"With intercept",input:"([1, 2, 3, 4], [3, 5, 7, 9], 0.1, 1000)",expected:"(2.0, 1.0)",hidden:!1},{id:"3",description:"Different slope",input:"([0, 1, 2, 3], [1, 4, 7, 10], 0.1, 1000)",expected:"(3.0, 1.0)",hidden:!0}],hints:["First compute predictions: y_pred = w * X + b","Then compute gradients: dw = (2/n) * sum((y_pred - y) * X)","Update weights: w = w - learning_rate * dw"],solution:`import numpy as np

def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.
    """
    w = 0.0
    b = 0.0
    n = len(X)

    for _ in range(iterations):
        # Forward pass
        y_pred = w * X + b

        # Compute gradients
        dw = (2/n) * np.sum((y_pred - y) * X)
        db = (2/n) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return (round(w, 2), round(b, 2))
`},{id:"logistic-regression",title:"Sigmoid Function",section:"supervised-learning",difficulty:"easy",description:`
## Sigmoid Function

Implement the sigmoid activation function used in logistic regression.

### Formula
\`\`\`
sigmoid(x) = 1 / (1 + exp(-x))
\`\`\`

### Properties
- Output is always between 0 and 1
- sigmoid(0) = 0.5
- Monotonically increasing

### Function Signature
\`\`\`python
def sigmoid(x: np.ndarray) -> np.ndarray:
\`\`\`
    `,examples:[{input:"np.array([0, 1, -1])",output:"[0.5, 0.731059, 0.268941]",explanation:"sigmoid(0)=0.5, sigmoid(1)≈0.73, sigmoid(-1)≈0.27"}],starterCode:`import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with sigmoid applied element-wise
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Zero input",input:"[0]",expected:"[0.5]",hidden:!1},{id:"2",description:"Positive and negative",input:"bool(np.allclose(sigmoid(np.array([-1, 0, 1])), [0.268941, 0.5, 0.731059], atol=1e-5))",expected:"True",hidden:!1},{id:"3",description:"Large values",input:"bool(np.allclose(sigmoid(np.array([-10, 10])), [4.5e-05, 0.999955], atol=1e-5))",expected:"True",hidden:!0}],hints:["Use np.exp() for the exponential function.","The formula is 1 / (1 + exp(-x))"],solution:`import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))
`},{id:"decision-tree-split",title:"Gini Impurity",section:"supervised-learning",difficulty:"medium",description:`
## Gini Impurity

Calculate the Gini impurity for a set of labels. This metric is used in decision trees to determine the best split.

### Formula
\`\`\`
Gini = 1 - sum(p_i^2)
\`\`\`

Where p_i is the proportion of class i in the set.

### Properties
- Gini = 0 means pure (all same class)
- Gini = 0.5 means maximum impurity (for binary classification with equal split)

### Function Signature
\`\`\`python
def gini_impurity(labels: np.ndarray) -> float:
\`\`\`
    `,examples:[{input:"np.array([0, 0, 0, 0])",output:"0.0",explanation:"All same class, pure node"},{input:"np.array([0, 0, 1, 1])",output:"0.5",explanation:"Equal split, maximum impurity for binary"}],starterCode:`import numpy as np

def gini_impurity(labels: np.ndarray) -> float:
    """
    Calculate Gini impurity for a set of labels.

    Args:
        labels: Array of class labels (integers)

    Returns:
        Gini impurity value between 0 and 1
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Pure node",input:"[0, 0, 0, 0]",expected:"0.0",hidden:!1},{id:"2",description:"Maximum impurity",input:"[0, 0, 1, 1]",expected:"0.5",hidden:!1},{id:"3",description:"Unequal split",input:"[0, 0, 0, 1]",expected:"0.375",hidden:!0}],hints:["Count the occurrences of each class.","Calculate the proportion (probability) of each class.","Gini = 1 - sum of squared probabilities."],solution:`import numpy as np

def gini_impurity(labels: np.ndarray) -> float:
    """
    Calculate Gini impurity for a set of labels.
    """
    if len(labels) == 0:
        return 0.0

    # Count occurrences of each class
    _, counts = np.unique(labels, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(labels)

    # Gini = 1 - sum(p^2)
    return 1 - np.sum(probabilities ** 2)
`},{id:"logistic-regression-full",title:"Logistic Regression with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
## Logistic Regression with Gradient Descent

Implement binary logistic regression from scratch.

### Model
\`\`\`
z = X @ w + b
y_pred = sigmoid(z) = 1 / (1 + exp(-z))
\`\`\`

### Loss (Binary Cross-Entropy)
\`\`\`
L = -1/m * sum(y * log(y_pred) + (1-y) * log(1-y_pred))
\`\`\`

### Gradients
\`\`\`
dw = 1/m * X.T @ (y_pred - y)
db = 1/m * sum(y_pred - y)
\`\`\`
    `,examples:[{input:"X (100, 2), y binary labels, 1000 iterations",output:"Trained weights and bias",explanation:"Learns decision boundary separating classes"}],starterCode:`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    """
    Train logistic regression using gradient descent.

    Args:
        X: Features (m samples, n features)
        y: Binary labels (m,)
        learning_rate: Step size
        iterations: Number of iterations

    Returns:
        w: Learned weights (n,)
        b: Learned bias (scalar)
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    # Your code here
    pass

    return np.round(w, 4), round(b, 4)
`,testCases:[{id:"1",description:"Simple separable data",input:"(lambda r: bool(np.allclose(r[0], [6.0141, 6.0141], atol=1e-3) and np.isclose(r[1], -9.1984, atol=1e-3)))(logistic_regression(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1]), 0.5, 1000))",expected:"True",hidden:!1}],hints:["Forward: z = X @ w + b, then y_pred = sigmoid(z)","Gradients: dw = (1/m) * X.T @ (y_pred - y)","Update: w = w - lr * dw, b = b - lr * db"],solution:`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(iterations):
        # Forward pass
        z = X @ w + b
        y_pred = sigmoid(z)

        # Compute gradients
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return np.round(w, 4), round(b, 4)
`},{id:"binary-cross-entropy",title:"Binary Cross-Entropy Loss",section:"supervised-learning",difficulty:"easy",description:`
## Binary Cross-Entropy Loss

Implement the binary cross-entropy (log loss) function.

### Formula
\`\`\`
BCE = -1/m * sum(y * log(p) + (1-y) * log(1-p))
\`\`\`

Where:
- y: True labels (0 or 1)
- p: Predicted probabilities
- m: Number of samples

### Numerical Stability
Clip predictions to avoid log(0):
\`\`\`python
p = np.clip(p, 1e-15, 1 - 1e-15)
\`\`\`
    `,examples:[{input:"y = [1, 0, 1], p = [0.9, 0.1, 0.8]",output:"0.1446",explanation:"Low loss for confident correct predictions"}],starterCode:`import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities

    Returns:
        loss: Scalar BCE loss
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect predictions",input:"([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0])",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"([1, 0, 1], [0.9, 0.1, 0.8])",expected:"0.1446",hidden:!1}],hints:["Clip predictions for numerical stability","Apply the formula element-wise","Take the mean over all samples"],solution:`import numpy as np

def binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip for numerical stability
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # BCE formula
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return round(loss, 4)
`}],x2=[{id:"kmeans-clustering",title:"K-Means: Assign to Nearest Centroid",section:"unsupervised-learning",difficulty:"medium",description:`
## K-Means: Assign Points to Nearest Centroid

Implement the assignment step of K-means clustering: given data points and centroids, assign each point to its nearest centroid.

### Function Signature
\`\`\`python
def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
\`\`\`

### Args
- X: Data points of shape (n_samples, n_features)
- centroids: Cluster centers of shape (k, n_features)

### Returns
- Array of cluster assignments (integers 0 to k-1)

### Distance
Use Euclidean distance: sqrt(sum((x - c)^2))
    `,examples:[{input:"X = [[0, 0], [1, 1], [10, 10]], centroids = [[0, 0], [10, 10]]",output:"[0, 0, 1]",explanation:"First two points closer to centroid 0, last point closer to centroid 1"}],starterCode:`import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to its nearest centroid.

    Args:
        X: Data points of shape (n_samples, n_features)
        centroids: Cluster centers of shape (k, n_features)

    Returns:
        Array of cluster assignments (shape: n_samples)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple 2D case",input:"([[0, 0], [1, 1], [10, 10]], [[0, 0], [10, 10]])",expected:"[0, 0, 1]",hidden:!1},{id:"2",description:"Three clusters",input:"([[0, 0], [5, 5], [10, 10]], [[0, 0], [5, 5], [10, 10]])",expected:"[0, 1, 2]",hidden:!1},{id:"3",description:"Tiebreaker (first centroid wins)",input:"([[5, 5]], [[0, 0], [10, 10]])",expected:"[0]",hidden:!0}],hints:["Calculate distance from each point to each centroid.","Use np.linalg.norm() for Euclidean distance.","Use np.argmin() to find the closest centroid."],solution:`import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to its nearest centroid.
    """
    # Calculate distances from each point to each centroid
    # Using broadcasting: X[:, np.newaxis] has shape (n, 1, d)
    # centroids has shape (k, d)
    # Difference has shape (n, k, d)
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    # Return index of minimum distance for each point
    return np.argmin(distances, axis=1)
`},{id:"pca-implementation",title:"PCA: Center Data",section:"unsupervised-learning",difficulty:"easy",description:`
## PCA: Center the Data

The first step in PCA is to center the data by subtracting the mean of each feature.

### Function Signature
\`\`\`python
def center_data(X: np.ndarray) -> np.ndarray:
\`\`\`

### Args
- X: Data matrix of shape (n_samples, n_features)

### Returns
- Centered data where each column has mean 0

### Why Center?
Centering ensures PCA finds directions of maximum variance from the origin, not from an arbitrary point.
    `,examples:[{input:"X = [[1, 2], [3, 4], [5, 6]]",output:"[[-2, -2], [0, 0], [2, 2]]",explanation:"Column means are [3, 4], so we subtract them"}],starterCode:`import numpy as np

def center_data(X: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean of each feature.

    Args:
        X: Data matrix of shape (n_samples, n_features)

    Returns:
        Centered data matrix of same shape
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic centering",input:"[[1, 2], [3, 4], [5, 6]]",expected:"[[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]]",hidden:!1},{id:"2",description:"Already centered",input:"[[-1, -1], [0, 0], [1, 1]]",expected:"[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]",hidden:!1},{id:"3",description:"Single column",input:"[[10], [20], [30]]",expected:"[[-10.0], [0.0], [10.0]]",hidden:!0}],hints:["Calculate the mean of each column using np.mean() with axis=0.","Subtract the mean from the data (broadcasting handles this)."],solution:`import numpy as np

def center_data(X: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean of each feature.
    """
    mean = np.mean(X, axis=0)
    return X - mean
`}],v2=[{id:"perceptron",title:"ReLU Activation",section:"deep-learning",difficulty:"easy",description:`
## ReLU Activation Function

Implement the Rectified Linear Unit (ReLU) activation function, one of the most popular activation functions in deep learning.

### Formula
\`\`\`
ReLU(x) = max(0, x)
\`\`\`

### Properties
- Output is x if x > 0, else 0
- Helps with vanishing gradient problem
- Computationally efficient

### Function Signature
\`\`\`python
def relu(x: np.ndarray) -> np.ndarray:
\`\`\`
    `,examples:[{input:"np.array([-2, -1, 0, 1, 2])",output:"[0, 0, 0, 1, 2]",explanation:"Negative values become 0, positive values unchanged"}],starterCode:`import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function.

    Args:
        x: Input array of any shape

    Returns:
        Array with ReLU applied element-wise
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Mixed values",input:"[-2, -1, 0, 1, 2]",expected:"[0, 0, 0, 1, 2]",hidden:!1},{id:"2",description:"All negative",input:"[-5, -3, -1]",expected:"[0, 0, 0]",hidden:!1},{id:"3",description:"All positive",input:"[1, 2, 3]",expected:"[1, 2, 3]",hidden:!0}],hints:["Use np.maximum() to compare with 0.","Alternatively, use boolean indexing or np.where()."],solution:`import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function.
    """
    return np.maximum(0, x)
`},{id:"neural-network-forward",title:"Dense Layer Forward Pass",section:"deep-learning",difficulty:"medium",description:`
## Dense Layer Forward Pass

Implement the forward pass of a fully-connected (dense) neural network layer.

### Formula
\`\`\`
output = activation(X @ W + b)
\`\`\`

Where:
- X: Input of shape (batch_size, input_dim)
- W: Weights of shape (input_dim, output_dim)
- b: Bias of shape (output_dim,)

### Function Signature
\`\`\`python
def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
\`\`\`

Use ReLU as the activation function.
    `,examples:[{input:"X = [[1, 2]], W = [[1, 0], [0, 1]], b = [1, 1]",output:"[[2, 3]]",explanation:"X @ W = [[1, 2]], + b = [[2, 3]], ReLU keeps positive"}],starterCode:`import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute forward pass of a dense layer with ReLU activation.

    Args:
        X: Input of shape (batch_size, input_dim)
        W: Weights of shape (input_dim, output_dim)
        b: Bias of shape (output_dim,)

    Returns:
        Output of shape (batch_size, output_dim)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic forward pass",input:"([[1, 2]], [[1, 0], [0, 1]], [1, 1])",expected:"[[2, 3]]",hidden:!1},{id:"2",description:"With negative pre-activation",input:"([[1, 1]], [[1, -2], [-1, 1]], [0, 0])",expected:"[[0, 0]]",hidden:!1},{id:"3",description:"Batch of 2",input:"([[1, 0], [0, 1]], [[2, 0], [0, 2]], [0, 0])",expected:"[[2, 0], [0, 2]]",hidden:!0}],hints:["First compute the linear transformation: Z = X @ W + b","Then apply ReLU: output = max(0, Z)"],solution:`import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute forward pass of a dense layer with ReLU activation.
    """
    # Linear transformation
    Z = np.dot(X, W) + b
    # ReLU activation
    return np.maximum(0, Z)
`},{id:"backpropagation",title:"Softmax Function",section:"deep-learning",difficulty:"medium",description:`
## Softmax Function

Implement the softmax function, used to convert raw scores to probabilities in multi-class classification.

### Formula
\`\`\`
softmax(x)_i = exp(x_i) / sum(exp(x_j))
\`\`\`

### Properties
- Output sums to 1
- All outputs are positive
- Larger inputs get larger probabilities

### Function Signature
\`\`\`python
def softmax(x: np.ndarray) -> np.ndarray:
\`\`\`

**Note**: For numerical stability, subtract max(x) before computing exp.
    `,examples:[{input:"np.array([1, 2, 3])",output:"[0.090031, 0.244728, 0.665241]",explanation:"Higher values get higher probabilities, sum = 1"}],starterCode:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.

    Args:
        x: Input array (1D)

    Returns:
        Array of same shape with softmax probabilities
    """
    # Your code here
    # Hint: subtract max for numerical stability
    pass
`,testCases:[{id:"1",description:"Basic softmax",input:"bool(np.allclose(softmax(np.array([1, 2, 3])), [0.090031, 0.244728, 0.665241], atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Equal inputs",input:"bool(np.allclose(softmax(np.array([1, 1, 1])), [0.333333, 0.333333, 0.333333], atol=1e-5))",expected:"True",hidden:!1},{id:"3",description:"Large values",input:"bool(np.allclose(softmax(np.array([100, 101, 102])), [0.090031, 0.244728, 0.665241], atol=1e-5))",expected:"True",hidden:!0}],hints:["Subtract the max value for numerical stability.","Compute exp of each element.","Divide by the sum of all exp values."],solution:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)
`}],_2=[{id:"precision-recall-f1",title:"Precision and Recall",section:"model-evaluation",difficulty:"medium",description:`
## Precision and Recall

Calculate precision and recall from true labels and predictions.

### Formulas
\`\`\`
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
\`\`\`

Where:
- TP = True Positives (predicted 1, actual 1)
- FP = False Positives (predicted 1, actual 0)
- FN = False Negatives (predicted 0, actual 1)

### Function Signature
\`\`\`python
def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
\`\`\`

Returns (precision, recall) rounded to 4 decimal places.
    `,examples:[{input:"y_true = [1, 1, 0, 1, 0], y_pred = [1, 0, 0, 1, 1]",output:"(0.6667, 0.6667)",explanation:"TP=2, FP=1, FN=1. Precision=2/3, Recall=2/3"}],starterCode:`import numpy as np

def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate precision and recall for binary classification.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Tuple of (precision, recall) rounded to 4 decimal places
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Mixed predictions",input:"([1, 1, 0, 1, 0], [1, 0, 0, 1, 1])",expected:"(0.6667, 0.6667)",hidden:!1},{id:"2",description:"Perfect predictions",input:"([1, 1, 0, 0], [1, 1, 0, 0])",expected:"(1.0, 1.0)",hidden:!1},{id:"3",description:"High precision, low recall",input:"([1, 1, 1, 1, 0], [1, 0, 0, 0, 0])",expected:"(1.0, 0.25)",hidden:!0}],hints:["Count TP where both y_true and y_pred are 1.","Count FP where y_pred is 1 but y_true is 0.","Count FN where y_true is 1 but y_pred is 0."],solution:`import numpy as np

def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate precision and recall for binary classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate TP, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return (round(precision, 4), round(recall, 4))
`},{id:"cross-validation",title:"K-Fold Split Indices",section:"model-evaluation",difficulty:"medium",description:`
## K-Fold Cross-Validation Indices

Generate train and validation indices for K-fold cross-validation.

### Concept
Split data into K equal parts. For each fold:
- Use 1 part for validation
- Use remaining K-1 parts for training

### Function Signature
\`\`\`python
def kfold_indices(n_samples: int, k: int) -> list:
\`\`\`

Returns list of (train_indices, val_indices) tuples for each fold.
    `,examples:[{input:"n_samples = 6, k = 3",output:"[([2,3,4,5], [0,1]), ([0,1,4,5], [2,3]), ([0,1,2,3], [4,5])]",explanation:"Split [0,1,2,3,4,5] into 3 folds of size 2"}],starterCode:`import numpy as np

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.

    Args:
        n_samples: Total number of samples
        k: Number of folds

    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic 3-fold",input:"(6, 3)",expected:"[[[2, 3, 4, 5], [0, 1]], [[0, 1, 4, 5], [2, 3]], [[0, 1, 2, 3], [4, 5]]]",hidden:!1},{id:"2",description:"2-fold",input:"(4, 2)",expected:"[[[2, 3], [0, 1]], [[0, 1], [2, 3]]]",hidden:!1},{id:"3",description:"4-fold",input:"(8, 4)",expected:"[[[2, 3, 4, 5, 6, 7], [0, 1]], [[0, 1, 4, 5, 6, 7], [2, 3]], [[0, 1, 2, 3, 6, 7], [4, 5]], [[0, 1, 2, 3, 4, 5], [6, 7]]]",hidden:!0}],hints:["Divide indices into k equal-sized chunks.","For each fold, one chunk is validation, rest are training.","Use np.array_split() to create chunks."],solution:`import numpy as np

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.
    """
    indices = np.arange(n_samples)
    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        val_indices = list(range(current, current + fold_size))
        train_indices = [i for i in range(n_samples) if i not in val_indices]
        folds.append((train_indices, val_indices))
        current += fold_size

    return folds
`},{id:"confusion-matrix",title:"Accuracy Score",section:"model-evaluation",difficulty:"easy",description:`
## Accuracy Score

Calculate the accuracy of predictions - the proportion of correct predictions.

### Formula
\`\`\`
Accuracy = (Number of correct predictions) / (Total predictions)
\`\`\`

### Function Signature
\`\`\`python
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
\`\`\`

### Note
While simple, accuracy can be misleading for imbalanced datasets.
    `,examples:[{input:"y_true = [0, 1, 1, 0, 1], y_pred = [0, 1, 0, 0, 1]",output:"0.8",explanation:"4 correct out of 5 predictions"}],starterCode:`import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic accuracy",input:"([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])",expected:"0.8",hidden:!1},{id:"2",description:"Perfect accuracy",input:"([1, 1, 0, 0], [1, 1, 0, 0])",expected:"1.0",hidden:!1},{id:"3",description:"Zero accuracy",input:"([0, 0, 1, 1], [1, 1, 0, 0])",expected:"0.0",hidden:!0}],hints:["Compare y_true and y_pred element-wise.","Count matches and divide by total.","np.mean() of boolean array gives proportion of True."],solution:`import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)
`}],w2=[{id:"mlp-forward-backward",title:"MLP Forward & Backward Pass",section:"neural-networks",difficulty:"hard",description:"\n## MLP Forward & Backward Pass\n\nImplement both forward and backward passes for a 2-layer MLP (Multi-Layer Perceptron).\n\n### Architecture\n```\nInput (n_features) → Hidden (n_hidden) → Output (n_classes)\n```\n\n### Forward Pass\n```\nZ1 = X @ W1 + b1\nA1 = ReLU(Z1)\nZ2 = A1 @ W2 + b2\noutput = softmax(Z2)\n```\n\nReturn: `(output, cache)` where cache = `(X, Z1, A1, Z2, W1, W2)`\n\n### Backward Pass (Backpropagation)\nUsing the chain rule, gradients flow backwards:\n\n```\nLoss → dZ2 → dW2, db2 → dA1 → dZ1 (ReLU) → dW1, db1\n```\n\n**Key Formulas:**\n- `dZ2 = output - Y_onehot` (softmax + cross-entropy derivative)\n- `dW2 = (1/m) * A1.T @ dZ2`\n- `db2 = (1/m) * sum(dZ2, axis=0)`\n- `dZ1 = (dZ2 @ W2.T) * (Z1 > 0)` (ReLU derivative)\n- `dW1 = (1/m) * X.T @ dZ1`\n- `db1 = (1/m) * sum(dZ1, axis=0)`\n\nReturn: `{'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...}`\n    ",examples:[{input:"X shape (2, 3), hidden_size=4, output_size=2",output:"Forward: probabilities (2, 2), rows sum to 1. Backward: gradients for all weights/biases",explanation:"Forward computes predictions; backward computes gradients for training"}],starterCode:`import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mlp_forward(X, W1, b1, W2, b2):
    """
    Forward pass for a 2-layer MLP.

    Args:
        X: Input data (batch_size, n_features)
        W1: First layer weights (n_features, n_hidden)
        b1: First layer bias (n_hidden,)
        W2: Second layer weights (n_hidden, n_classes)
        b2: Second layer bias (n_classes,)

    Returns:
        output: Class probabilities (batch_size, n_classes)
        cache: Tuple (X, Z1, A1, Z2, W1, W2) for backprop
    """
    # Your code here
    pass

def mlp_backward(Y, output, cache):
    """
    Backward pass (backpropagation) for 2-layer MLP.

    Args:
        Y: One-hot encoded labels (batch_size, n_classes)
        output: Predicted probabilities from forward pass
        cache: Cached values from forward pass (X, Z1, A1, Z2, W1, W2)

    Returns:
        grads: Dictionary with dW1, db1, dW2, db2
    """
    X, Z1, A1, Z2, W1, W2 = cache
    m = X.shape[0]  # batch size

    # Your code here
    pass
`,testCases:[{id:"1",description:"Forward: output shape correct",input:"mlp_forward(np.array([[1, 2, 3], [4, 5, 6]]), np.ones((3, 4)), np.zeros(4), np.ones((4, 2)), np.zeros(2))[0].shape",expected:"(2, 2)",hidden:!1},{id:"2",description:"Forward: probabilities sum to 1",input:"round(float(np.sum(mlp_forward(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.zeros(2), np.array([[1, 0], [0, 1]]), np.zeros(2))[0][0])), 1)",expected:"1.0",hidden:!1},{id:"3",description:"Backward: dW2 shape matches W2",input:`(lambda: (
            X := np.array([[1.0, 2.0], [3.0, 4.0]]),
            W1 := np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            b1 := np.zeros(3),
            W2 := np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            b2 := np.zeros(2),
            result := mlp_forward(X, W1, b1, W2, b2),
            output := result[0],
            cache := result[1],
            Y := np.array([[1, 0], [0, 1]]),
            grads := mlp_backward(Y, output, cache),
            grads['dW2'].shape == W2.shape
        )[-1])()`,expected:"True",hidden:!1},{id:"4",description:"Backward: gradients are non-zero",input:`(lambda: (
            X := np.array([[1.0, 2.0], [3.0, 4.0]]),
            W1 := np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            b1 := np.zeros(3),
            W2 := np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            b2 := np.zeros(2),
            result := mlp_forward(X, W1, b1, W2, b2),
            output := result[0],
            cache := result[1],
            Y := np.array([[1, 0], [0, 1]]),
            grads := mlp_backward(Y, output, cache),
            bool(np.any(grads['dW1'] != 0) and np.any(grads['dW2'] != 0))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"Backward: all gradient keys present",input:`(lambda: (
            X := np.array([[1.0, 2.0]]),
            W1 := np.array([[0.1, 0.2], [0.3, 0.4]]),
            b1 := np.zeros(2),
            W2 := np.array([[0.1], [0.2]]),
            b2 := np.zeros(1),
            result := mlp_forward(X, W1, b1, W2, b2),
            grads := mlp_backward(np.array([[1]]), result[0], result[1]),
            sorted(grads.keys())
        )[-1])()`,expected:'["dW1", "dW2", "db1", "db2"]',hidden:!0}],hints:["Forward: Compute Z1 = X @ W1 + b1, then A1 = ReLU(Z1)","Forward: Compute Z2 = A1 @ W2 + b2, then output = softmax(Z2)","Forward: Store cache = (X, Z1, A1, Z2, W1, W2) for backprop","Backward: Start from output layer: dZ2 = output - Y","Backward: dW2 = (1/m) * A1.T @ dZ2, db2 = (1/m) * sum(dZ2, axis=0)","Backward: Propagate through ReLU: dZ1 = (dZ2 @ W2.T) * (Z1 > 0)","Backward: dW1 = (1/m) * X.T @ dZ1, db1 = (1/m) * sum(dZ1, axis=0)"],solution:`import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mlp_forward(X, W1, b1, W2, b2):
    # First layer
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Second layer
    Z2 = A1 @ W2 + b2
    output = softmax(Z2)

    # Cache for backprop
    cache = (X, Z1, A1, Z2, W1, W2)

    return output, cache

def mlp_backward(Y, output, cache):
    X, Z1, A1, Z2, W1, W2 = cache
    m = X.shape[0]

    # Output layer gradient (softmax + cross-entropy)
    dZ2 = output - Y
    dW2 = (1/m) * A1.T @ dZ2
    db2 = (1/m) * np.sum(dZ2, axis=0)

    # Hidden layer gradient
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
    dW1 = (1/m) * X.T @ dZ1
    db1 = (1/m) * np.sum(dZ1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
`},{id:"cross-entropy-loss",title:"Cross-Entropy Loss",section:"neural-networks",difficulty:"easy",description:`
## Cross-Entropy Loss

Implement the cross-entropy loss function for multi-class classification.

### Formula
\`\`\`
L = -1/m * sum(Y * log(Y_pred))
\`\`\`

Where:
- Y is one-hot encoded true labels
- Y_pred is predicted probabilities
- m is batch size

### Numerical Stability
Add small epsilon to avoid log(0):
\`\`\`python
np.log(Y_pred + 1e-15)
\`\`\`
    `,examples:[{input:"Y_pred = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], Y = [[1,0,0], [0,1,0]]",output:"0.2231",explanation:"-1/2 * (log(0.7) + log(0.8))"}],starterCode:`import numpy as np

def cross_entropy_loss(Y_pred, Y_true):
    """
    Compute cross-entropy loss.

    Args:
        Y_pred: Predicted probabilities (batch_size, n_classes)
        Y_true: One-hot encoded true labels (batch_size, n_classes)

    Returns:
        loss: Scalar cross-entropy loss
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect prediction",input:"abs(cross_entropy_loss(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1, 0], [0, 1]])))",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"cross_entropy_loss(np.array([[0.7, 0.3], [0.2, 0.8]]), np.array([[1, 0], [0, 1]]))",expected:"0.2899",hidden:!1}],hints:["Use np.log with a small epsilon for numerical stability","Multiply element-wise: Y_true * np.log(Y_pred)","Sum and negate, then divide by batch size"],solution:`import numpy as np

def cross_entropy_loss(Y_pred, Y_true):
    m = Y_pred.shape[0]
    epsilon = 1e-15
    log_probs = np.log(Y_pred + epsilon)
    loss = -np.sum(Y_true * log_probs) / m
    return round(loss, 4)
`},{id:"weight-init",title:"Xavier/He Weight Initialization",section:"neural-networks",difficulty:"medium",description:`
## Weight Initialization

Implement Xavier and He weight initialization to prevent vanishing/exploding gradients.

### Xavier Initialization (for tanh/sigmoid)
\`\`\`
W = randn(n_in, n_out) * sqrt(2 / (n_in + n_out))
\`\`\`

### He Initialization (for ReLU)
\`\`\`
W = randn(n_in, n_out) * sqrt(2 / n_in)
\`\`\`

### Why It Matters
- Keeps variance of activations stable across layers
- Prevents gradients from vanishing or exploding
- Enables training of deep networks
    `,examples:[{input:'n_in=784, n_out=256, method="he"',output:"Weights with std ≈ 0.0505",explanation:"sqrt(2/784) ≈ 0.0505"}],starterCode:`import numpy as np

def initialize_weights(n_in, n_out, method='xavier'):
    """
    Initialize weights using Xavier or He initialization.

    Args:
        n_in: Number of input units
        n_out: Number of output units
        method: 'xavier' or 'he'

    Returns:
        std: Standard deviation of initialized weights (rounded to 4 decimals)
    """
    np.random.seed(42)  # For reproducibility
    # Your code here
    pass
`,testCases:[{id:"1",description:"Xavier std correct",input:'(100, 50, "xavier")',expected:"0.1151",hidden:!1},{id:"2",description:"He std correct",input:'(100, 50, "he")',expected:"0.1409",hidden:!1}],hints:["Use np.random.randn to generate random numbers","Xavier: multiply by sqrt(2 / (n_in + n_out))","He: multiply by sqrt(2 / n_in)"],solution:`import numpy as np

def initialize_weights(n_in, n_out, method='xavier'):
    np.random.seed(42)

    if method == 'xavier':
        std = np.sqrt(2.0 / (n_in + n_out))
    elif method == 'he':
        std = np.sqrt(2.0 / n_in)
    else:
        raise ValueError("Method must be 'xavier' or 'he'")

    W = np.random.randn(n_in, n_out) * std
    return round(np.std(W), 4)
`},{id:"batch-norm",title:"Batch Normalization",section:"neural-networks",difficulty:"medium",description:`
## Batch Normalization

Implement batch normalization to stabilize training.

### Forward Pass
\`\`\`
1. Compute batch mean: μ = mean(x, axis=0)
2. Compute batch variance: σ² = var(x, axis=0)
3. Normalize: x_norm = (x - μ) / sqrt(σ² + ε)
4. Scale and shift: out = γ * x_norm + β
\`\`\`

### Return Format
Return \`(out, cache)\` where:
- \`out\`: Normalized and scaled output
- \`cache\`: Tuple \`(X, X_norm, mu, var, gamma, eps)\` for backward pass

### Benefits
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization
    `,examples:[{input:"X with mean=5, var=4, gamma=1, beta=0",output:"Normalized X with mean≈0, var≈1",explanation:"BatchNorm normalizes each feature"}],starterCode:`import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Batch normalization forward pass.

    Args:
        X: Input (batch_size, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant for numerical stability

    Returns:
        out: Normalized output (batch_size, features)
        cache: Tuple (X, X_norm, mu, var, gamma, eps) for backward pass
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output mean near zero",input:"bool(np.allclose(np.mean(batch_norm_forward(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.ones(2), np.zeros(2))[0], axis=0), 0, atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Gamma scales output",input:"list(np.round(batch_norm_forward(np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([2.0, 3.0]), np.zeros(2))[0][1], 1))",expected:"[2.0, 3.0]",hidden:!0}],hints:["Compute mean along axis=0 (batch dimension)","Compute variance along axis=0","Normalize: (X - mean) / sqrt(var + eps)","Apply scale and shift: gamma * normalized + beta"],solution:`import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    # Compute batch statistics
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    # Normalize
    X_norm = (X - mu) / np.sqrt(var + eps)

    # Scale and shift
    out = gamma * X_norm + beta

    # Cache for backward pass
    cache = (X, X_norm, mu, var, gamma, eps)

    return out, cache
`},{id:"dropout",title:"Dropout",section:"neural-networks",difficulty:"easy",description:`
## Dropout Regularization

Implement dropout to prevent overfitting.

### Training Mode
\`\`\`
1. Generate mask: mask = (rand < keep_prob) / keep_prob
2. Apply mask: out = x * mask
\`\`\`

### Key Points
- Randomly "drop" neurons with probability (1 - keep_prob)
- Scale by 1/keep_prob to maintain expected value
- Disabled during inference (return input unchanged)
    `,examples:[{input:"X = [[1, 2, 3, 4]], keep_prob=0.5",output:"Some values zeroed, others scaled by 2",explanation:"Half neurons dropped, rest doubled"}],starterCode:`import numpy as np

def dropout_forward(X, keep_prob=0.5, training=True):
    """
    Apply dropout to input.

    Args:
        X: Input array
        keep_prob: Probability of keeping a neuron
        training: Whether in training mode

    Returns:
        out: Output after dropout
        mask: Dropout mask (for backward pass)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Inference returns unchanged",input:"dropout_forward(np.array([[1, 2, 3, 4]]), 0.5, False)[0].tolist()",expected:"[[1, 2, 3, 4]]",hidden:!1},{id:"2",description:"Expected value preserved",input:`(lambda: (
            np.random.seed(42),
            X := np.ones((1000, 100)),
            out := dropout_forward(X, 0.8, True)[0],
            bool(abs(np.mean(out) - 1.0) < 0.1)
        )[-1])()`,expected:"True",hidden:!0}],hints:["In training: create binary mask with np.random.rand","Scale mask by 1/keep_prob (inverted dropout)","In inference: return input unchanged"],solution:`import numpy as np

def dropout_forward(X, keep_prob=0.5, training=True):
    if not training:
        return X, None

    # Create mask and scale
    mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
    out = X * mask

    return out, mask
`}],b2=[{id:"conv2d-forward",title:"2D Convolution",section:"cnn",difficulty:"hard",description:`
## 2D Convolution Operation

Implement the forward pass of a 2D convolution (no padding, stride=1).

### Operation
For each position (i, j) in the output:
\`\`\`
out[i, j] = sum(input[i:i+kH, j:j+kW] * kernel)
\`\`\`

### Output Size
\`\`\`
out_height = input_height - kernel_height + 1
out_width = input_width - kernel_width + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d(image, kernel):
    # image: (H, W)
    # kernel: (kH, kW)
    # output: (H-kH+1, W-kW+1)
\`\`\`
    `,examples:[{input:"image 4x4, kernel 3x3",output:"output 2x2",explanation:"4-3+1 = 2 in each dimension"}],starterCode:`import numpy as np

def conv2d(image, kernel):
    """
    Apply 2D convolution to an image.

    Args:
        image: Input image (H, W)
        kernel: Convolution kernel (kH, kW)

    Returns:
        output: Convolved image (H-kH+1, W-kW+1)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Identity kernel",input:"([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]])",expected:"[[5]]",hidden:!1},{id:"2",description:"Edge detection",input:"([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])",expected:"[[5, 5], [5, 5]]",hidden:!1}],hints:["Use nested loops to slide the kernel over the image","At each position, compute element-wise product and sum","Output size is (H-kH+1, W-kW+1)"],solution:`import numpy as np

def conv2d(image, kernel):
    image = np.array(image)
    kernel = np.array(kernel)

    H, W = image.shape
    kH, kW = kernel.shape

    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i:i+kH, j:j+kW]
            output[i, j] = np.sum(region * kernel)

    return output.astype(int).tolist()
`},{id:"max-pool",title:"Max Pooling",section:"cnn",difficulty:"medium",description:`
## Max Pooling

Implement 2x2 max pooling with stride 2.

### Operation
Divide input into non-overlapping 2x2 regions and take maximum of each.

### Output Size
\`\`\`
out_height = input_height // 2
out_width = input_width // 2
\`\`\`

### Purpose
- Reduces spatial dimensions
- Provides translation invariance
- Reduces computation
    `,examples:[{input:"[[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]",output:"[[6, 8], [14, 16]]",explanation:"Max of each 2x2 region"}],starterCode:`import numpy as np

def max_pool2d(image, pool_size=2):
    """
    Apply 2D max pooling.

    Args:
        image: Input image (H, W)
        pool_size: Size of pooling window

    Returns:
        output: Pooled image (H//pool_size, W//pool_size)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"4x4 to 2x2",input:"[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",expected:"[[6, 8], [14, 16]]",hidden:!1},{id:"2",description:"With negative values",input:"[[-1, -2, -3, -4], [-5, -6, -7, -8], [-9, -10, -11, -12], [-13, -14, -15, -16]]",expected:"[[-1, -3], [-9, -11]]",hidden:!0}],hints:["Iterate with step size = pool_size","For each 2x2 region, use np.max()","Output dimensions are input dimensions // pool_size"],solution:`import numpy as np

def max_pool2d(image, pool_size=2):
    image = np.array(image)
    H, W = image.shape

    out_H = H // pool_size
    out_W = W // pool_size

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i*pool_size:(i+1)*pool_size,
                          j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)

    return output.astype(int).tolist()
`},{id:"flatten-layer",title:"Flatten Layer",section:"cnn",difficulty:"easy",description:`
## Flatten Layer

Implement the flatten operation that converts a 3D feature map to a 1D vector for the fully connected layer.

### Operation
\`\`\`
(batch, height, width, channels) → (batch, height * width * channels)
\`\`\`

### Usage
- Connects convolutional layers to fully connected layers
- Preserves batch dimension
    `,examples:[{input:"shape (2, 4, 4, 3)",output:"shape (2, 48)",explanation:"4 * 4 * 3 = 48 features per sample"}],starterCode:`import numpy as np

def flatten(X):
    """
    Flatten feature maps to vectors.

    Args:
        X: Input tensor (batch, height, width, channels)

    Returns:
        output: Flattened tensor (batch, height*width*channels)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct output shape",input:"flatten(np.random.randn(2, 4, 4, 3)).shape",expected:"(2, 48)",hidden:!1},{id:"2",description:"Values preserved",input:"bool(np.allclose(flatten(np.arange(24).reshape(1, 2, 3, 4)), np.arange(24).reshape(1, 24)))",expected:"True",hidden:!0}],hints:["Get batch size as X.shape[0]","Use reshape to flatten remaining dimensions","np.reshape(X, (batch_size, -1)) uses -1 to infer size"],solution:`import numpy as np

def flatten(X):
    batch_size = X.shape[0]
    return X.reshape(batch_size, -1)
`},{id:"conv-output-size",title:"Convolution Output Size",section:"cnn",difficulty:"easy",description:`
## Calculate Convolution Output Size

Implement a function to calculate the output dimensions of a convolution layer.

### Formula
\`\`\`
output_size = (input_size - kernel_size + 2 * padding) / stride + 1
\`\`\`

### Parameters
- **input_size**: Height or width of input
- **kernel_size**: Height or width of kernel
- **padding**: Zero-padding added to input
- **stride**: Step size of kernel
    `,examples:[{input:"input=32, kernel=3, padding=1, stride=1",output:"32",explanation:"(32 - 3 + 2*1) / 1 + 1 = 32"}],starterCode:`def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    """
    Calculate output size of a convolution layer.

    Args:
        input_size: Input dimension (height or width)
        kernel_size: Kernel dimension
        padding: Zero-padding
        stride: Stride

    Returns:
        output_size: Output dimension
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Same padding",input:"(32, 3, 1, 1)",expected:"32",hidden:!1},{id:"2",description:"No padding, stride 2",input:"(32, 3, 0, 2)",expected:"15",hidden:!1},{id:"3",description:"Large kernel",input:"(224, 7, 3, 2)",expected:"112",hidden:!0}],hints:["Apply the formula: (input - kernel + 2*padding) / stride + 1","Use integer division (//) for the result"],solution:`def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    return (input_size - kernel_size + 2 * padding) // stride + 1
`},{id:"conv2d-advanced",title:"Advanced 2D Convolution",section:"cnn",difficulty:"hard",description:`
## Advanced 2D Convolution

Implement a full-featured 2D convolution with **padding**, **stride**, and **groups** support.

### Parameters
- **padding**: Zero-padding added to input borders
- **stride**: Step size when sliding the kernel
- **groups**: Split input/output channels into groups (used in depthwise separable convolutions)

### Group Convolution
When \`groups > 1\`:
- Input channels are split into \`groups\` chunks
- Output channels are split into \`groups\` chunks
- Each group's input is convolved only with its corresponding kernel group
- \`in_channels\` and \`out_channels\` must be divisible by \`groups\`

### Special Cases
- \`groups=1\`: Standard convolution
- \`groups=in_channels=out_channels\`: Depthwise convolution (MobileNet)

### Output Size Formula
\`\`\`
H_out = (H_in + 2*padding - kernel_height) // stride + 1
W_out = (W_in + 2*padding - kernel_width) // stride + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    # input: (batch, in_channels, H, W)
    # kernel: (out_channels, in_channels//groups, kH, kW)
    # output: (batch, out_channels, H_out, W_out)
\`\`\`
    `,examples:[{input:"input (1, 4, 5, 5), kernel (8, 2, 3, 3), groups=2",output:"output (1, 8, 3, 3)",explanation:"Groups=2: channels 0-1 use kernels 0-3, channels 2-3 use kernels 4-7"},{input:"input (1, 1, 5, 5), kernel (1, 1, 3, 3), padding=1, stride=2",output:"output (1, 1, 3, 3)",explanation:"(5+2*1-3)//2+1 = 3"}],starterCode:`import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    """
    Advanced 2D convolution with padding, stride, and groups.

    Args:
        input: Input tensor (batch, in_channels, H, W)
        kernel: Convolution kernels (out_channels, in_channels//groups, kH, kW)
        padding: Zero-padding added to input (default: 0)
        stride: Stride of the convolution (default: 1)
        groups: Number of groups for grouped convolution (default: 1)

    Returns:
        output: Convolved tensor (batch, out_channels, H_out, W_out)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic convolution with padding and stride",input:"conv2d_advanced(np.ones((1, 1, 5, 5)), np.ones((1, 1, 3, 3)), 1, 2, 1).shape",expected:"(1, 1, 3, 3)",hidden:!1},{id:"2",description:"Grouped convolution (groups=2)",input:"conv2d_advanced(np.arange(32).reshape(1, 2, 4, 4).astype(float), np.ones((4, 1, 2, 2)), 0, 1, 2).shape",expected:"(1, 4, 3, 3)",hidden:!1},{id:"3",description:"Depthwise convolution (groups=in_channels)",input:"conv2d_advanced(np.ones((2, 3, 4, 4)), np.ones((3, 1, 2, 2)), 0, 1, 3).shape",expected:"(2, 3, 3, 3)",hidden:!0},{id:"4",description:"Verify correct output values with groups",input:"bool(np.allclose(conv2d_advanced(np.ones((1, 2, 3, 3)), np.ones((2, 1, 2, 2)), 0, 1, 2), np.array([[[[4., 4.], [4., 4.]], [[4., 4.], [4., 4.]]]])) )",expected:"True",hidden:!0}],hints:["First, pad the input using np.pad() along the H and W dimensions","Calculate output dimensions: H_out = (H_padded - kH) // stride + 1","For groups: split input channels into chunks of size in_channels//groups","For groups: split kernels into chunks of size out_channels//groups","Convolve each input group with its corresponding kernel group","Concatenate results along the channel dimension"],solution:`import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    input = np.array(input, dtype=float)
    kernel = np.array(kernel, dtype=float)

    batch, in_channels, H, W = input.shape
    out_channels, kernel_in_channels, kH, kW = kernel.shape

    # Pad input
    if padding > 0:
        input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    _, _, H_padded, W_padded = input.shape

    # Calculate output dimensions
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    # Initialize output
    output = np.zeros((batch, out_channels, H_out, W_out))

    # Channels per group
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    for g in range(groups):
        # Input channels for this group
        in_start = g * in_channels_per_group
        in_end = in_start + in_channels_per_group

        # Output channels for this group
        out_start = g * out_channels_per_group
        out_end = out_start + out_channels_per_group

        # Get the input slice for this group
        input_group = input[:, in_start:in_end, :, :]

        # Get the kernels for this group
        kernel_group = kernel[out_start:out_end, :, :, :]

        # Perform convolution for this group
        for b in range(batch):
            for oc in range(out_channels_per_group):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        region = input_group[b, :, h_start:h_start+kH, w_start:w_start+kW]
                        output[b, out_start + oc, i, j] = np.sum(region * kernel_group[oc])

    return output
`}],k2=[{id:"scaled-dot-product-attention",title:"Scaled Dot-Product Attention",section:"transformers",difficulty:"medium",description:`
## Scaled Dot-Product Attention

Implement the core attention mechanism from "Attention Is All You Need".

### Formula
\`\`\`
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
\`\`\`

Where:
- Q: Queries (seq_len, d_k)
- K: Keys (seq_len, d_k)
- V: Values (seq_len, d_v)
- d_k: Dimension of keys

### Steps
1. Compute attention scores: Q @ K.T
2. Scale by sqrt(d_k)
3. Apply softmax
4. Multiply by V
    `,examples:[{input:"Q, K, V all (4, 8) - 4 tokens, 8 dims",output:"output (4, 8)",explanation:"Each token attends to all tokens"}],starterCode:`import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries (seq_len, d_k)
        K: Keys (seq_len, d_k)
        V: Values (seq_len, d_v)

    Returns:
        output: Attention output (seq_len, d_v)
        attention_weights: Attention weights (seq_len, seq_len)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape correct",input:"scaled_dot_product_attention(np.random.randn(4, 8), np.random.randn(4, 8), np.random.randn(4, 8))[0].shape",expected:"(4, 8)",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"bool(np.allclose(scaled_dot_product_attention(np.ones((4, 8)), np.ones((4, 8)), np.ones((4, 8)))[1].sum(axis=-1), 1.0))",expected:"True",hidden:!1}],hints:["Compute scores = Q @ K.T","Scale by sqrt(d_k) where d_k = Q.shape[-1]","Apply softmax along the last axis","Return softmax(scores) @ V"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Apply softmax
    attention_weights = softmax(scores)

    # Compute output
    output = attention_weights @ V

    return output, attention_weights
`},{id:"multi-head-attention",title:"Multi-Head Attention",section:"transformers",difficulty:"hard",description:`
## Multi-Head Attention

Implement multi-head attention from the Transformer architecture.

### Concept
\`\`\`
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
\`\`\`

### Steps
1. Project Q, K, V with different learned projections for each head
2. Apply scaled dot-product attention in parallel
3. Concatenate heads
4. Project back to original dimension
    `,examples:[{input:"X (4, 64), num_heads=8, d_model=64",output:"output (4, 64)",explanation:"8 heads with d_k=8 each, concatenated to 64"}],starterCode:`import numpy as np

def multi_head_attention(Q, K, V, num_heads):
    """
    Multi-head attention mechanism.

    Args:
        Q: Queries (seq_len, d_model)
        K: Keys (seq_len, d_model)
        V: Values (seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        output: Multi-head attention output (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"multi_head_attention(np.random.randn(4, 64), np.random.randn(4, 64), np.random.randn(4, 64), 8).shape",expected:"(4, 64)",hidden:!1}],hints:["Reshape Q, K, V to (seq_len, num_heads, d_k)","Apply attention for each head","Reshape back to (seq_len, d_model)"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_attention(Q, K, V, num_heads):
    seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Reshape to (seq_len, num_heads, d_k)
    Q = Q.reshape(seq_len, num_heads, d_k)
    K = K.reshape(seq_len, num_heads, d_k)
    V = V.reshape(seq_len, num_heads, d_k)

    # Transpose to (num_heads, seq_len, d_k)
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    # Scaled dot-product attention for each head
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    attention = softmax(scores)
    heads = attention @ V

    # Transpose and reshape back
    heads = heads.transpose(1, 0, 2)  # (seq_len, num_heads, d_k)
    output = heads.reshape(seq_len, d_model)

    return output
`},{id:"positional-encoding",title:"Sinusoidal Positional Encoding",section:"transformers",difficulty:"medium",description:`
## Sinusoidal Positional Encoding

Implement the positional encoding from the original Transformer paper.

### Formulas
\`\`\`
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
\`\`\`

Where:
- pos: Position in sequence
- i: Dimension index
- d_model: Model dimension

### Purpose
- Injects position information into embeddings
- Allows model to learn relative positions
- Fixed (not learned) encodings
    `,examples:[{input:"max_len=100, d_model=512",output:"PE matrix (100, 512)",explanation:"Position encoding for 100 positions"}],starterCode:`import numpy as np

def positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        PE: Positional encoding matrix (max_len, d_model)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct shape",input:"positional_encoding(50, 64).shape",expected:"(50, 64)",hidden:!1},{id:"2",description:"First position starts with sin(0)=0",input:"bool(np.isclose(positional_encoding(10, 4)[0, 0], 0.0))",expected:"True",hidden:!0}],hints:["Create position indices: np.arange(max_len)","Create dimension indices for the divisor","Use div_term = 10000^(2i/d_model)","Apply sin to even indices, cos to odd indices"],solution:`import numpy as np

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
`},{id:"layer-norm",title:"Layer Normalization",section:"transformers",difficulty:"easy",description:`
## Layer Normalization

Implement layer normalization, used in Transformers instead of batch normalization.

### Formula
\`\`\`
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
\`\`\`

### Difference from BatchNorm
- **BatchNorm**: Normalizes across batch dimension
- **LayerNorm**: Normalizes across feature dimension
- LayerNorm works better for sequence models
    `,examples:[{input:"X (batch, seq_len, features)",output:"Normalized X, each position normalized independently",explanation:"Normalize across last dimension"}],starterCode:`import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    """
    Apply layer normalization.

    Args:
        X: Input (batch, seq_len, features) or (seq_len, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant

    Returns:
        output: Normalized tensor
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output mean near zero",input:"bool(np.allclose(layer_norm(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), np.ones(4), np.zeros(4)).mean(axis=-1), 0.0, atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Gamma and beta work",input:"bool(np.allclose(layer_norm(np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([1.0, 2.0, 1.0, 2.0]), np.zeros(4)), np.array([[-1.0, -2.0, 1.0, 2.0]])))",expected:"True",hidden:!0}],hints:["Compute mean and variance along the last axis","Use keepdims=True for proper broadcasting","Apply normalization: (x - mean) / sqrt(var + eps)"],solution:`import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    # Compute statistics along last dimension
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)

    # Scale and shift
    return gamma * X_norm + beta
`},{id:"causal-mask",title:"Causal Attention Mask",section:"transformers",difficulty:"easy",description:`
## Causal (Autoregressive) Mask

Create a causal mask for decoder self-attention to prevent attending to future tokens.

### Mask Structure
\`\`\`
[[0, -inf, -inf, -inf],
 [0,    0, -inf, -inf],
 [0,    0,    0, -inf],
 [0,    0,    0,    0]]
\`\`\`

### Usage
- Add mask to attention scores before softmax
- -inf becomes 0 after softmax
- Ensures each position only attends to previous positions
    `,examples:[{input:"seq_len=4",output:"Lower triangular mask (4, 4)",explanation:"Position i can only attend to positions <= i"}],starterCode:`import numpy as np

def create_causal_mask(seq_len):
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Causal mask (seq_len, seq_len)
               0 for allowed positions, -inf for masked
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct shape and pattern",input:"create_causal_mask(4).shape",expected:"(4, 4)",hidden:!1},{id:"2",description:"Lower triangular zeros, upper triangular -inf",input:'bool(np.allclose(create_causal_mask(3), np.array([[0.0, float("-inf"), float("-inf")], [0.0, 0.0, float("-inf")], [0.0, 0.0, 0.0]])))',expected:"True",hidden:!0}],hints:["Create a matrix of ones using np.ones((seq_len, seq_len))","Use np.triu to get upper triangular matrix (excluding diagonal)","Replace 1s with -inf"],solution:`import numpy as np

def create_causal_mask(seq_len):
    # Create upper triangular matrix (above diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    # Replace 1s with -inf
    mask = np.where(mask == 1, float('-inf'), 0.0)
    return mask
`}],S2=[{id:"vae-reparameterization",title:"VAE Reparameterization Trick",section:"generative-models",difficulty:"medium",description:`
## VAE Reparameterization Trick

Implement the reparameterization trick that allows backpropagation through stochastic sampling in VAEs.

### The Problem
We want to sample z ~ N(μ, σ²), but sampling is not differentiable.

### The Solution
\`\`\`
ε ~ N(0, 1)
z = μ + σ * ε
\`\`\`

### Why It Works
- ε is sampled independently of parameters
- z is now a deterministic function of μ, σ, and ε
- Gradients can flow through μ and σ
    `,examples:[{input:"mu = [0, 1], log_var = [0, 0]",output:"z = mu + exp(0.5 * log_var) * epsilon",explanation:"log_var=0 means σ=1"}],starterCode:`import numpy as np

def reparameterize(mu, log_var):
    """
    Sample from latent distribution using reparameterization trick.

    Args:
        mu: Mean of latent distribution (batch, latent_dim)
        log_var: Log variance of latent distribution (batch, latent_dim)

    Returns:
        z: Sampled latent vectors (batch, latent_dim)
    """
    np.random.seed(42)  # For reproducibility
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape matches input",input:"reparameterize(np.zeros((2, 4)), np.zeros((2, 4))).shape",expected:"(2, 4)",hidden:!1},{id:"2",description:"Zero variance case has correct shape",input:"reparameterize(np.ones((3, 5)), np.full((3, 5), -1000)).shape",expected:"(3, 5)",hidden:!0}],hints:["std = exp(0.5 * log_var)","Sample epsilon from standard normal","z = mu + std * epsilon"],solution:`import numpy as np

def reparameterize(mu, log_var):
    np.random.seed(42)
    # Compute standard deviation
    std = np.exp(0.5 * log_var)
    # Sample epsilon from standard normal
    eps = np.random.randn(*mu.shape)
    # Reparameterization trick
    z = mu + std * eps
    return z
`},{id:"vae-loss",title:"VAE Loss Function",section:"generative-models",difficulty:"medium",description:`
## VAE Loss (ELBO)

Implement the VAE loss function: reconstruction loss + KL divergence.

### Loss Components
\`\`\`
L = L_reconstruction + L_KL

L_reconstruction = MSE(x, x_reconstructed)
L_KL = -0.5 * sum(1 + log_var - mu² - exp(log_var))
\`\`\`

### Intuition
- **Reconstruction loss**: Output should match input
- **KL divergence**: Latent distribution should be close to N(0,1)
    `,examples:[{input:"Perfect reconstruction, mu=0, var=1",output:"Loss ≈ 0",explanation:"Both terms are minimized"}],starterCode:`import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    """
    Compute VAE loss (negative ELBO).

    Args:
        x: Original input (batch, features)
        x_reconstructed: Reconstructed input (batch, features)
        mu: Latent mean (batch, latent_dim)
        log_var: Latent log variance (batch, latent_dim)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect case",input:"(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 2)), np.zeros((2, 2)))",expected:"(0.0, 0.0, 0.0)",hidden:!1},{id:"2",description:"Non-zero KL",input:"(np.zeros((1, 4)), np.zeros((1, 4)), np.ones((1, 2)), np.zeros((1, 2)))",expected:"(0.3161, 0.0, 0.3161)",hidden:!0}],hints:["Reconstruction: np.mean((x - x_reconstructed)²)","KL: -0.5 * sum(1 + log_var - mu² - exp(log_var))","Average over batch"],solution:`import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    # Reconstruction loss (MSE)
    recon_loss = np.mean((x - x_reconstructed) ** 2)

    # KL divergence
    kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

    total_loss = recon_loss + kl_loss

    return round(total_loss, 4), round(recon_loss, 4), round(kl_loss, 4)
`},{id:"diffusion-noise-schedule",title:"Diffusion Noise Schedule",section:"generative-models",difficulty:"easy",description:`
## Diffusion Noise Schedule

Implement a linear noise schedule for diffusion models.

### Linear Schedule
\`\`\`
β_t = β_start + t * (β_end - β_start) / T

where t = 0, 1, ..., T-1
\`\`\`

### Derived Quantities
\`\`\`
α_t = 1 - β_t
ᾱ_t = prod(α_1, ..., α_t)  # cumulative product
\`\`\`

These control how much noise is added at each step.
    `,examples:[{input:"T=1000, beta_start=0.0001, beta_end=0.02",output:"betas, alphas, alpha_bars arrays",explanation:"Standard DDPM schedule"}],starterCode:`import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    """
    Create linear noise schedule for diffusion.

    Args:
        T: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        betas: Beta values (T,)
        alphas: Alpha values (T,)
        alpha_bars: Cumulative alpha product (T,)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Beta range correct",input:"linear_noise_schedule(100, 0.0001, 0.02)[0].shape",expected:"(100,)",hidden:!1},{id:"2",description:"Alpha bar decreases - last < first",input:"bool(linear_noise_schedule(50, 0.001, 0.01)[2][-1] < linear_noise_schedule(50, 0.001, 0.01)[2][0])",expected:"True",hidden:!0}],hints:["Use np.linspace for linear interpolation","alphas = 1 - betas","alpha_bars = np.cumprod(alphas)"],solution:`import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    # Linear schedule
    betas = np.linspace(beta_start, beta_end, T)

    # Compute alphas
    alphas = 1 - betas

    # Cumulative product
    alpha_bars = np.cumprod(alphas)

    return betas, alphas, alpha_bars
`},{id:"diffusion-forward",title:"Diffusion Forward Process",section:"generative-models",difficulty:"medium",description:`
## Diffusion Forward Process (Adding Noise)

Implement the forward diffusion process that adds noise to data.

### Formula
\`\`\`
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

where ε ~ N(0, I)
\`\`\`

### Intuition
- As t increases, ᾱ_t decreases
- More noise is added, signal is reduced
- At t=T, x_T ≈ pure noise
    `,examples:[{input:"x_0 (image), t=500, T=1000",output:"x_t (noisy image)",explanation:"Halfway through diffusion process"}],starterCode:`import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    """
    Add noise to data using forward diffusion.

    Args:
        x_0: Original data (batch, ...)
        t: Timestep (int)
        alpha_bars: Cumulative alpha products (T,)

    Returns:
        x_t: Noisy data at timestep t
        noise: The noise that was added
    """
    np.random.seed(42)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape matches input",input:"diffusion_forward(np.ones((2, 4)), 50, np.linspace(0.99, 0.01, 100))[0].shape",expected:"(2, 4)",hidden:!1},{id:"2",description:"Noise shape matches input",input:"diffusion_forward(np.ones((3, 5)), 10, np.linspace(0.99, 0.01, 100))[1].shape",expected:"(3, 5)",hidden:!0}],hints:["Get alpha_bar_t = alpha_bars[t]","Sample noise from standard normal","Apply formula: sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise"],solution:`import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    np.random.seed(42)

    alpha_bar_t = alpha_bars[t]

    # Sample noise
    noise = np.random.randn(*x_0.shape)

    # Forward process
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * noise

    return x_t, noise
`},{id:"vqvae-quantization",title:"VQ-VAE Vector Quantization",section:"generative-models",difficulty:"hard",description:`
## VQ-VAE Vector Quantization

Implement the vector quantization layer used in VQ-VAE (Vector Quantized Variational Autoencoder).

### Overview
Unlike standard VAEs that use continuous latent spaces, VQ-VAE uses **discrete** latent representations by mapping encoder outputs to the nearest embedding in a learned codebook.

### Quantization Process
1. **Encoder output**: z_e of shape (batch, H, W, D)
2. **Codebook**: K embedding vectors of dimension D
3. **Find nearest**: For each spatial position, find the closest codebook entry
4. **Quantize**: Replace z_e with the nearest codebook vector z_q

### Distance Calculation
\`\`\`
distances[b, h, w, k] = ||z_e[b, h, w] - codebook[k]||²
\`\`\`

### Straight-Through Estimator
During training, gradients flow through z_q to z_e by copying gradients:
\`\`\`
z_q = z_e + stop_gradient(z_q - z_e)
\`\`\`
(We don't implement this here, just the forward pass)

### VQ-VAE Loss Components
\`\`\`
L = reconstruction + β * ||sg[z_e] - e||² + ||z_e - sg[e]||²
\`\`\`
- **Codebook loss**: Moves embeddings toward encoder outputs
- **Commitment loss**: Commits encoder to embeddings

### Function Signature
\`\`\`python
def vq_quantize(z_e, codebook):
    # z_e: (batch, H, W, embedding_dim) - encoder output
    # codebook: (num_embeddings, embedding_dim) - K embedding vectors
    # Returns: z_q, indices
\`\`\`
    `,examples:[{input:"z_e (2, 4, 4, 64), codebook (512, 64)",output:"z_q (2, 4, 4, 64), indices (2, 4, 4)",explanation:"Each spatial position mapped to one of 512 codes"}],starterCode:`import numpy as np

def vq_quantize(z_e, codebook):
    """
    Vector quantization for VQ-VAE.

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        codebook: Embedding vectors (num_embeddings, embedding_dim)

    Returns:
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        indices: Codebook indices used (batch, H, W)
    """
    # Your code here
    pass


def vq_loss(z_e, z_q, beta=0.25):
    """
    Compute VQ-VAE losses (codebook + commitment).

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        beta: Commitment loss weight (default: 0.25)

    Returns:
        codebook_loss: Loss to update codebook
        commitment_loss: Loss to commit encoder to codebook
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"z_q output shape matches input",input:"vq_quantize(np.random.randn(2, 4, 4, 8), np.random.randn(16, 8))[0].shape",expected:"(2, 4, 4, 8)",hidden:!1},{id:"2",description:"Indices shape is correct",input:"vq_quantize(np.random.randn(2, 4, 4, 8), np.random.randn(16, 8))[1].shape",expected:"(2, 4, 4)",hidden:!1},{id:"3",description:"Indices are valid codebook indices (all less than K)",input:"bool(np.all(vq_quantize(np.random.randn(2, 3, 3, 4), np.random.randn(8, 4))[1] < 8))",expected:"True",hidden:!0},{id:"4",description:"VQ loss returns tuple of two values",input:"len(vq_loss(np.random.randn(2, 3, 3, 4), np.random.randn(2, 3, 3, 4), 0.25))",expected:"2",hidden:!0}],hints:["Reshape z_e to (batch*H*W, D) for easier distance computation","Use broadcasting: ||a - b||² = ||a||² + ||b||² - 2*a·b","np.argmin along the codebook axis gives indices","Index into codebook with the indices to get z_q","Reshape back to original spatial dimensions","For loss: codebook_loss = ||z_e.detach() - z_q||², commitment = ||z_e - z_q.detach()||²"],solution:`import numpy as np

def vq_quantize(z_e, codebook):
    z_e = np.array(z_e)
    codebook = np.array(codebook)

    batch, H, W, D = z_e.shape
    K, _ = codebook.shape

    # Flatten spatial dimensions: (batch*H*W, D)
    z_flat = z_e.reshape(-1, D)

    # Compute distances using: ||a-b||² = ||a||² + ||b||² - 2*a·b
    # z_flat: (N, D), codebook: (K, D)
    z_sq = np.sum(z_flat ** 2, axis=1, keepdims=True)  # (N, 1)
    codebook_sq = np.sum(codebook ** 2, axis=1)        # (K,)
    cross = z_flat @ codebook.T                        # (N, K)

    distances = z_sq + codebook_sq - 2 * cross         # (N, K)

    # Find nearest codebook entry
    indices_flat = np.argmin(distances, axis=1)        # (N,)

    # Get quantized vectors
    z_q_flat = codebook[indices_flat]                  # (N, D)

    # Reshape back
    z_q = z_q_flat.reshape(batch, H, W, D)
    indices = indices_flat.reshape(batch, H, W)

    return z_q, indices


def vq_loss(z_e, z_q, beta=0.25):
    z_e = np.array(z_e)
    z_q = np.array(z_q)

    # Codebook loss: ||sg[z_e] - z_q||² (moves codebook toward encoder output)
    # In practice, sg[z_e] means z_e is treated as constant
    codebook_loss = np.mean((z_e - z_q) ** 2)

    # Commitment loss: ||z_e - sg[z_q]||² (commits encoder to codebook)
    # Same computation, but gradient only flows to z_e
    commitment_loss = beta * np.mean((z_e - z_q) ** 2)

    return round(codebook_loss, 4), round(commitment_loss, 4)
`},{id:"kl-divergence",title:"KL Divergence (Gaussians)",section:"generative-models",difficulty:"easy",description:`
## KL Divergence Between Gaussians

Compute the KL divergence between two univariate Gaussian distributions.

### Formula
\`\`\`
KL(P || Q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5
\`\`\`

Where P = N(μ_p, σ_p²) and Q = N(μ_q, σ_q²)

### Properties
- KL ≥ 0
- KL = 0 iff P = Q
- Not symmetric: KL(P||Q) ≠ KL(Q||P)
    `,examples:[{input:"P = N(0, 1), Q = N(0, 1)",output:"0",explanation:"Same distribution"}],starterCode:`import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute KL divergence between two Gaussians.

    Args:
        mu_p, sigma_p: Mean and std of P
        mu_q, sigma_q: Mean and std of Q

    Returns:
        kl: KL(P || Q)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Same distribution",input:"(0, 1, 0, 1)",expected:"0.0",hidden:!1},{id:"2",description:"Different means",input:"(1, 1, 0, 1)",expected:"0.5",hidden:!1},{id:"3",description:"Different variances",input:"(0, 2, 0, 1)",expected:"0.8069",hidden:!0}],hints:["Apply the formula directly","Use np.log for natural logarithm","Remember: σ² is variance, σ is std"],solution:`import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    term3 = -0.5

    kl = term1 + term2 + term3
    return round(kl, 4)
`}],C2=[{id:"numpy-array-creation",title:"Array Creation Methods",section:"numpy-fundamentals",difficulty:"easy",description:`
## NumPy Array Creation

Implement a function that creates various NumPy arrays using different methods.

### Requirements
Create and return a dictionary with:
- \`zeros\`: 3x4 array of zeros
- \`ones\`: 2x3 array of ones
- \`arange\`: array from 0 to 9
- \`linspace\`: 5 evenly spaced values from 0 to 1
- \`eye\`: 4x4 identity matrix

### Function Signature
\`\`\`python
def create_arrays() -> dict:
\`\`\`
    `,examples:[{input:"create_arrays()",output:"{'zeros': array([[0,0,0,0],...]), ...}",explanation:"Dictionary with 5 different array types"}],starterCode:`import numpy as np

def create_arrays() -> dict:
    """
    Create various NumPy arrays.

    Returns:
        Dictionary with keys: zeros, ones, arange, linspace, eye
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Zeros shape correct",input:'create_arrays()["zeros"].shape',expected:"(3, 4)",hidden:!1},{id:"2",description:"Linspace length correct",input:'len(create_arrays()["linspace"])',expected:"5",hidden:!1},{id:"3",description:"Eye is identity",input:'bool(np.allclose(create_arrays()["eye"], np.eye(4)))',expected:"True",hidden:!0}],hints:["Use np.zeros((3, 4)) for zero array","np.linspace(start, stop, num) creates evenly spaced values","np.eye(n) creates an identity matrix"],solution:`import numpy as np

def create_arrays() -> dict:
    return {
        'zeros': np.zeros((3, 4)),
        'ones': np.ones((2, 3)),
        'arange': np.arange(10),
        'linspace': np.linspace(0, 1, 5),
        'eye': np.eye(4)
    }
`},{id:"numpy-indexing",title:"Advanced Indexing",section:"numpy-fundamentals",difficulty:"medium",description:`
## NumPy Advanced Indexing

Implement various indexing operations on a 2D array.

### Given
\`\`\`python
arr = np.arange(20).reshape(4, 5)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19]])
\`\`\`

### Requirements
Return a dictionary with:
- \`row_1\`: Second row (index 1)
- \`col_2\`: Third column (index 2)
- \`subarray\`: Rows 1-2, columns 2-4
- \`diagonal\`: Main diagonal
- \`reversed\`: Array with rows reversed

### Function Signature
\`\`\`python
def advanced_indexing(arr: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"arr = np.arange(20).reshape(4, 5)",output:"{'row_1': [5,6,7,8,9], ...}",explanation:"Various slicing and indexing operations"}],starterCode:`import numpy as np

def advanced_indexing(arr: np.ndarray) -> dict:
    """
    Perform advanced indexing operations.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with indexed results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Row extraction",input:'advanced_indexing(np.arange(20).reshape(4, 5))["row_1"].tolist()',expected:"[5, 6, 7, 8, 9]",hidden:!1},{id:"2",description:"Subarray correct",input:'advanced_indexing(np.arange(20).reshape(4, 5))["subarray"].tolist()',expected:"[[7, 8, 9], [12, 13, 14]]",hidden:!1}],hints:["arr[1] or arr[1, :] gets row 1","arr[:, 2] gets column 2","arr[1:3, 2:5] gets a subarray","np.diag(arr) extracts diagonal","arr[::-1] reverses rows"],solution:`import numpy as np

def advanced_indexing(arr: np.ndarray) -> dict:
    return {
        'row_1': arr[1],
        'col_2': arr[:, 2],
        'subarray': arr[1:3, 2:5],
        'diagonal': np.diag(arr),
        'reversed': arr[::-1]
    }
`},{id:"numpy-broadcasting",title:"Broadcasting Operations",section:"numpy-fundamentals",difficulty:"medium",description:`
## NumPy Broadcasting

Implement operations that leverage NumPy's broadcasting rules.

### Task
Given a 2D array of shape (3, 4):
1. Subtract the row means from each row (center rows)
2. Subtract the column means from each column (center columns)
3. Add a 1D bias vector of shape (4,) to each row
4. Multiply by a column vector of shape (3, 1)

### Broadcasting Rules
- Arrays are compatible when dimensions are equal or one of them is 1
- Operations proceed element-wise on matching dimensions

### Function Signature
\`\`\`python
def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"arr (3,4), bias (4,), scale (3,1)",output:"{'row_centered': ..., 'col_centered': ..., 'biased': ..., 'scaled': ...}",explanation:"Various broadcasting operations"}],starterCode:`import numpy as np

def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
    """
    Perform broadcasting operations.

    Args:
        arr: 2D array of shape (3, 4)
        bias: 1D array of shape (4,)
        scale: 2D array of shape (3, 1)

    Returns:
        Dictionary with results of broadcasting operations
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Row centering",input:'bool(np.allclose(broadcasting_ops(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]), np.zeros(4), np.ones((3,1)))["row_centered"].mean(axis=1), 0))',expected:"True",hidden:!1},{id:"2",description:"Bias addition shape",input:'broadcasting_ops(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]), np.array([1,1,1,1]), np.ones((3,1)))["biased"].shape',expected:"(3, 4)",hidden:!1}],hints:["Row means: arr.mean(axis=1, keepdims=True)","Column means: arr.mean(axis=0, keepdims=True) or arr.mean(axis=0)","Bias shape (4,) broadcasts with arr shape (3, 4)","Scale shape (3, 1) broadcasts with arr shape (3, 4)"],solution:`import numpy as np

def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
    # Row centered: subtract mean of each row
    row_means = arr.mean(axis=1, keepdims=True)  # (3, 1)
    row_centered = arr - row_means

    # Column centered: subtract mean of each column
    col_means = arr.mean(axis=0)  # (4,)
    col_centered = arr - col_means

    # Add bias to each row
    biased = arr + bias  # bias (4,) broadcasts to (3, 4)

    # Scale each row
    scaled = arr * scale  # scale (3, 1) broadcasts to (3, 4)

    return {
        'row_centered': row_centered,
        'col_centered': col_centered,
        'biased': biased,
        'scaled': scaled
    }
`},{id:"numpy-aggregations",title:"Aggregation Functions",section:"numpy-fundamentals",difficulty:"easy",description:"\n## NumPy Aggregations\n\nImplement common aggregation operations along different axes.\n\n### Task\nGiven a 2D array, compute:\n- Global statistics (mean, std, min, max, sum)\n- Row-wise statistics (along axis=1)\n- Column-wise statistics (along axis=0)\n- Argmax and argmin (indices of max/min values)\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- **Global**: `'global_mean'`, `'global_std'`, `'global_min'`, `'global_max'`, `'global_sum'`\n- **Row-wise**: `'row_mean'`, `'row_sum'`\n- **Column-wise**: `'col_mean'`, `'col_sum'`\n- **Indices**: `'argmax'` (2D index tuple), `'argmin'` (2D index tuple)\n    ",examples:[{input:"np.array([[1, 2, 3], [4, 5, 6]])",output:"{'global_mean': 3.5, 'global_sum': 21, 'row_sum': [6, 15], 'col_mean': [2.5, 3.5, 4.5], 'argmax': (1, 2), ...}",explanation:"Global, row-wise, and column-wise aggregations"}],starterCode:`import numpy as np

def compute_aggregations(arr: np.ndarray) -> dict:
    """
    Compute aggregation statistics.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with aggregation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Global mean",input:'compute_aggregations(np.array([[1, 2, 3], [4, 5, 6]]))["global_mean"]',expected:"3.5",hidden:!1},{id:"2",description:"Row sums",input:'compute_aggregations(np.array([[1, 2, 3], [4, 5, 6]]))["row_sum"].tolist()',expected:"[6, 15]",hidden:!1}],hints:["arr.mean() computes global mean","arr.mean(axis=0) computes column means","arr.mean(axis=1) computes row means","np.argmax(arr) gives index of max in flattened array"],solution:`import numpy as np

def compute_aggregations(arr: np.ndarray) -> dict:
    return {
        'global_mean': arr.mean(),
        'global_std': arr.std(),
        'global_min': arr.min(),
        'global_max': arr.max(),
        'global_sum': arr.sum(),
        'row_mean': arr.mean(axis=1),
        'row_sum': arr.sum(axis=1),
        'col_mean': arr.mean(axis=0),
        'col_sum': arr.sum(axis=0),
        'argmax': np.unravel_index(arr.argmax(), arr.shape),
        'argmin': np.unravel_index(arr.argmin(), arr.shape),
    }
`},{id:"numpy-reshape-transpose",title:"Reshape and Transpose",section:"numpy-fundamentals",difficulty:"medium",description:`
## Reshape and Transpose Operations

Master array shape manipulation - essential for ML data pipelines.

### Task
Given a 1D array of 24 elements:
1. Reshape to (4, 6)
2. Reshape to (2, 3, 4)
3. Transpose the 2D version
4. Swap axes on the 3D version
5. Flatten back to 1D

### Important Concepts
- \`reshape\` changes shape without changing data order
- \`transpose\` swaps axes
- \`-1\` in reshape means "infer this dimension"

### Expected Return Format
Return a dictionary with these keys:
- \`'arr_2d'\`: Reshaped to (4, 6)
- \`'arr_3d'\`: Reshaped to (2, 3, 4)
- \`'arr_2d_transposed'\`: Transpose of arr_2d, shape (6, 4)
- \`'arr_3d_swapped'\`: arr_3d with axes swapped (2,1,0), shape (4, 3, 2)
- \`'arr_flat'\`: Flattened back to 1D, shape (24,)
    `,examples:[{input:"np.arange(24)",output:"{'arr_2d': shape (4,6), 'arr_3d': shape (2,3,4), 'arr_2d_transposed': shape (6,4), ...}",explanation:"Various reshape and transpose operations"}],starterCode:`import numpy as np

def reshape_transpose(arr: np.ndarray) -> dict:
    """
    Perform reshape and transpose operations.

    Args:
        arr: 1D array of 24 elements

    Returns:
        Dictionary with reshaped arrays
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"2D reshape shape",input:'reshape_transpose(np.arange(24))["arr_2d"].shape',expected:"(4, 6)",hidden:!1},{id:"2",description:"3D reshape shape",input:'reshape_transpose(np.arange(24))["arr_3d"].shape',expected:"(2, 3, 4)",hidden:!1}],hints:["arr.reshape(4, 6) or arr.reshape(4, -1)","arr.reshape(2, 3, 4) for 3D",".T or .transpose() for transpose",".flatten() or .ravel() for 1D"],solution:`import numpy as np

def reshape_transpose(arr: np.ndarray) -> dict:
    # 2D reshape
    arr_2d = arr.reshape(4, 6)

    # 3D reshape
    arr_3d = arr.reshape(2, 3, 4)

    # Transpose 2D
    arr_2d_T = arr_2d.T

    # Swap axes on 3D (swap axis 0 and 2)
    arr_3d_swapped = arr_3d.transpose(2, 1, 0)

    # Flatten
    arr_flat = arr_2d.flatten()

    return {
        'arr_2d': arr_2d,
        'arr_3d': arr_3d,
        'arr_2d_transposed': arr_2d_T,
        'arr_3d_swapped': arr_3d_swapped,
        'arr_flat': arr_flat
    }
`}],E2=[{id:"einsum-basics",title:"Einsum Fundamentals",section:"einsum",difficulty:"easy",description:`
## Einsum Fundamentals

Einstein summation notation provides a powerful way to express array operations.

### Syntax
\`\`\`python
np.einsum('subscripts', operands)
\`\`\`

### Basic Operations
- \`'i->'\`: Sum all elements (reduce)
- \`'ij->i'\`: Sum along columns (row sums)
- \`'ij->j'\`: Sum along rows (column sums)
- \`'ii->i'\`: Extract diagonal
- \`'ij->ji'\`: Transpose

### Task
Implement basic einsum operations on a 2D array.

### Expected Return Format
Return a dictionary with these keys:
- \`'sum_all'\`: Sum of all elements
- \`'row_sum'\`: Sum along each row
- \`'col_sum'\`: Sum along each column
- \`'transpose'\`: Transposed array
- \`'diagonal'\`: Diagonal elements (for square matrices, else None)
    `,examples:[{input:"np.array([[1,2],[3,4]])",output:"{'sum_all': 10, 'row_sum': [3,7], 'col_sum': [4,6], 'transpose': [[1,3],[2,4]], 'diagonal': [1,4]}",explanation:"Basic einsum operations on a 2x2 matrix"}],starterCode:`import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    """
    Perform basic einsum operations.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with einsum results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Sum all elements",input:'einsum_basics(np.array([[1, 2], [3, 4]]))["sum_all"]',expected:"10",hidden:!1},{id:"2",description:"Transpose",input:'bool(np.array_equal(einsum_basics(np.array([[1, 2], [3, 4]]))["transpose"], np.array([[1, 3], [2, 4]])))',expected:"True",hidden:!1}],hints:["'ij->' sums all elements","'ij->i' sums each row","'ij->ji' transposes","'ii->i' extracts diagonal (for square matrices)"],solution:`import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    return {
        'sum_all': np.einsum('ij->', arr),
        'row_sum': np.einsum('ij->i', arr),
        'col_sum': np.einsum('ij->j', arr),
        'transpose': np.einsum('ij->ji', arr),
        'diagonal': np.einsum('ii->i', arr) if arr.shape[0] == arr.shape[1] else None
    }
`},{id:"einsum-matrix-ops",title:"Matrix Operations with Einsum",section:"einsum",difficulty:"medium",description:`
## Matrix Operations with Einsum

Einsum can express matrix multiplication and more complex operations.

### Key Operations
- \`'ik,kj->ij'\`: Matrix multiplication (A @ B)
- \`'ij,ij->ij'\`: Element-wise multiplication (Hadamard)
- \`'ij,ij->'\`: Frobenius inner product (sum of element-wise product)
- \`'ij,j->i'\`: Matrix-vector multiplication
- \`'i,j->ij'\`: Outer product

### Task
Implement matrix operations using einsum.

### Expected Return Format
Return a dictionary with these keys:
- \`'matmul'\`: Matrix multiplication A @ B
- \`'matvec'\`: Matrix-vector multiplication A @ v
- \`'outer_product'\`: Outer product of v with itself
- \`'hadamard'\`: Element-wise A * A
- \`'frobenius'\`: Sum of A * A (Frobenius inner product)
- \`'trace'\`: Trace of A (if square, else None)
    `,examples:[{input:"A=[[1,2],[3,4]], B=[[5,6],[7,8]], v=[1,2]",output:"{'matmul': [[19,22],[43,50]], 'matvec': [5,11], 'outer_product': [[1,2],[2,4]], 'hadamard': [[1,4],[9,16]], 'frobenius': 30, 'trace': 5}",explanation:"Matrix operations via einsum"}],starterCode:`import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    """
    Perform matrix operations using einsum.

    Args:
        A: Matrix of shape (m, k)
        B: Matrix of shape (k, n)
        v: Vector of shape (k,)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Matrix multiplication",input:'bool(np.array_equal(einsum_matrix_ops(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]), np.array([1,2]))["matmul"], np.array([[19,22],[43,50]])))',expected:"True",hidden:!1},{id:"2",description:"Outer product",input:'bool(np.array_equal(einsum_matrix_ops(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]), np.array([1,2]))["outer_product"], np.array([[1,2],[2,4]])))',expected:"True",hidden:!1}],hints:["'ik,kj->ij' contracts over k (matrix multiply)","'ij,j->i' contracts vector with columns","'i,j->ij' creates outer product (no contraction)"],solution:`import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    return {
        'matmul': np.einsum('ik,kj->ij', A, B),
        'matvec': np.einsum('ij,j->i', A, v),
        'outer_product': np.einsum('i,j->ij', v, v),
        'hadamard': np.einsum('ij,ij->ij', A, A),  # A * A
        'frobenius': np.einsum('ij,ij->', A, A),   # sum(A * A)
        'trace': np.einsum('ii->', A) if A.shape[0] == A.shape[1] else None
    }
`},{id:"einsum-batch-ops",title:"Batch Operations with Einsum",section:"einsum",difficulty:"medium",description:`
## Batch Operations with Einsum

Einsum shines for batch operations - multiple matrices at once.

### Batch Matrix Operations
- \`'bij,bjk->bik'\`: Batch matrix multiplication
- \`'bij,bj->bi'\`: Batch matrix-vector multiplication
- \`'bij->bji'\`: Batch transpose

### Attention-style Operations
- \`'bqd,bkd->bqk'\`: Query-Key dot products (attention scores)
- \`'bqk,bkv->bqv'\`: Weighted sum of values

### Task
Implement batch operations common in deep learning.

### Expected Return Format
Return a dictionary with these keys:
- \`'scores'\`: Raw attention scores Q @ K.T per batch, shape (batch, seq_q, seq_k)
- \`'attention_weights'\`: Softmax of scaled scores, shape (batch, seq_q, seq_k)
- \`'output'\`: Weighted sum of values, shape (batch, seq_q, dim_v)
    `,examples:[{input:"Q (2,4,8), K (2,6,8), V (2,6,16)",output:"{'scores': shape (2,4,6), 'attention_weights': shape (2,4,6), 'output': shape (2,4,16)}",explanation:"Batch attention: scores from Q@K.T, then weighted sum of V"}],starterCode:`import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    """
    Perform batch operations using einsum.

    Args:
        Q: Queries of shape (batch, seq_q, dim)
        K: Keys of shape (batch, seq_k, dim)
        V: Values of shape (batch, seq_k, dim_v)

    Returns:
        Dictionary with batch operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Attention scores shape",input:'einsum_batch_ops(np.ones((2, 4, 8)), np.ones((2, 6, 8)), np.ones((2, 6, 16)))["scores"].shape',expected:"(2, 4, 6)",hidden:!1},{id:"2",description:"Output shape",input:'einsum_batch_ops(np.ones((2, 4, 8)), np.ones((2, 6, 8)), np.ones((2, 6, 16)))["output"].shape',expected:"(2, 4, 16)",hidden:!1}],hints:["'bqd,bkd->bqk' computes Q @ K.T for each batch","'bqk,bkv->bqv' applies attention weights to values","b is the batch dimension, preserved in output"],solution:`import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    # Q: (batch, seq_q, dim)
    # K: (batch, seq_k, dim)
    # V: (batch, seq_k, dim_v)

    # Attention scores: Q @ K.T for each batch
    # Result: (batch, seq_q, seq_k)
    scores = np.einsum('bqd,bkd->bqk', Q, K)

    # Scale scores
    d_k = Q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Softmax (simplified, along last axis)
    exp_scores = np.exp(scaled_scores - scaled_scores.max(axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # Result: (batch, seq_q, dim_v)
    output = np.einsum('bqk,bkv->bqv', attention_weights, V)

    return {
        'scores': scores,
        'attention_weights': attention_weights,
        'output': output
    }
`},{id:"einsum-advanced",title:"Advanced Einsum Patterns",section:"einsum",difficulty:"hard",description:`
## Advanced Einsum Patterns

Complex tensor operations for deep learning.

### Multi-Head Attention Pattern
\`\`\`python
# Split heads: (batch, seq, heads, dim)
# Attention per head: 'bhqd,bhkd->bhqk'
# Combine heads: 'bhqv->bqhv' then reshape
\`\`\`

### Bilinear Operations
\`\`\`python
# x.T @ W @ y: 'i,ijk,k->j'
# Batch bilinear: 'bi,ijk,bk->bj'
\`\`\`

### Task
Implement multi-head attention using einsum.
    `,examples:[{input:"Q, K, V with multiple heads",output:"Multi-head attention output",explanation:"Parallel attention across heads"}],starterCode:`import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    """
    Multi-head attention using einsum.

    Args:
        Q: Queries (batch, seq_q, d_model)
        K: Keys (batch, seq_k, d_model)
        V: Values (batch, seq_k, d_model)
        num_heads: Number of attention heads

    Returns:
        output: (batch, seq_q, d_model)
    """
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"multi_head_attention_einsum(np.ones((2, 4, 64)), np.ones((2, 4, 64)), np.ones((2, 4, 64)), 8).shape",expected:"(2, 4, 64)",hidden:!1},{id:"2",description:"Output is valid array",input:"not np.isnan(multi_head_attention_einsum(np.ones((2, 4, 64)), np.ones((2, 4, 64)), np.ones((2, 4, 64)), 8)).any()",expected:"True",hidden:!0}],hints:["Reshape Q, K, V to (batch, seq, num_heads, d_k)","Use 'bqhd,bkhd->bhqk' for attention scores per head","Use 'bhqk,bkhd->bqhd' for weighted values","Reshape back to (batch, seq, d_model)"],solution:`import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Reshape to (batch, seq, num_heads, d_k)
    Q = Q.reshape(batch, seq_q, num_heads, d_k)
    K = K.reshape(batch, seq_k, num_heads, d_k)
    V = V.reshape(batch, seq_k, num_heads, d_k)

    # Compute attention scores for all heads
    # (batch, seq_q, heads, d_k) x (batch, seq_k, heads, d_k) -> (batch, heads, seq_q, seq_k)
    scores = np.einsum('bqhd,bkhd->bhqk', Q, K) / np.sqrt(d_k)

    # Softmax
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # (batch, heads, seq_q, seq_k) x (batch, seq_k, heads, d_k) -> (batch, seq_q, heads, d_k)
    output = np.einsum('bhqk,bkhd->bqhd', attention, V)

    # Reshape back to (batch, seq_q, d_model)
    output = output.reshape(batch, seq_q, d_model)

    return output
`},{id:"einsum-vs-matmul",title:"Einsum vs Traditional Operations",section:"einsum",difficulty:"easy",description:"\n## Einsum vs Traditional Operations\n\nCompare einsum with equivalent NumPy operations.\n\n### Equivalences\n| Einsum | NumPy |\n|--------|-------|\n| `'ij->'` | `np.sum(A)` |\n| `'ij->i'` | `np.sum(A, axis=1)` |\n| `'ik,kj->ij'` | `A @ B` |\n| `'ij,ij->'` | `np.sum(A * B)` |\n| `'ij->ji'` | `A.T` |\n\n### Task\nVerify einsum produces same results as traditional operations.\n\n### Expected Return Format\nReturn a dictionary with:\n- `'results'`: Dict mapping operation names to {'einsum': ..., 'numpy': ...}\n- `'all_match'`: Boolean indicating if all einsum results match numpy equivalents\n    ",examples:[{input:"A=[[1,2,3],[4,5,6]], B=[[1,2],[3,4],[5,6]]",output:"{'results': {'sum': {...}, 'matmul': {...}, ...}, 'all_match': True}",explanation:"All einsum operations produce identical results to NumPy equivalents"}],starterCode:`import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Compare einsum with traditional NumPy operations.

    Args:
        A: First matrix (m, k)
        B: Second matrix (k, n)

    Returns:
        Dictionary with 'einsum' and 'numpy' results for comparison
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"All results match",input:'einsum_equivalence(np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]), np.array([[1.0,2.0],[3.0,4.0],[5.0,6.0]]))["all_match"]',expected:"True",hidden:!1},{id:"2",description:"Matrix multiplication matches",input:'bool(np.allclose(einsum_equivalence(np.array([[1.0,2.0],[3.0,4.0]]), np.array([[5.0,6.0],[7.0,8.0]]))["results"]["matmul"]["einsum"], np.array([[19.0,22.0],[43.0,50.0]])))',expected:"True",hidden:!1}],hints:["Use np.allclose() to compare floating point arrays","Remember 'ik,kj->ij' is matrix multiplication"],solution:`import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    results = {
        'sum': {
            'einsum': np.einsum('ij->', A),
            'numpy': np.sum(A)
        },
        'row_sum': {
            'einsum': np.einsum('ij->i', A),
            'numpy': np.sum(A, axis=1)
        },
        'matmul': {
            'einsum': np.einsum('ik,kj->ij', A, B),
            'numpy': A @ B
        },
        'transpose': {
            'einsum': np.einsum('ij->ji', A),
            'numpy': A.T
        },
        'hadamard_sum': {
            'einsum': np.einsum('ij,ij->', A, A),
            'numpy': np.sum(A * A)
        }
    }

    # Verify all match
    all_match = all(
        np.allclose(v['einsum'], v['numpy'])
        for v in results.values()
    )

    return {'results': results, 'all_match': all_match}
`}],z2=[{id:"tensor-creation",title:"Tensor Creation (NumPy Style)",section:"pytorch-basics",difficulty:"easy",description:"\n## Tensor Creation\n\nLearn tensor creation patterns used in PyTorch, implemented with NumPy.\n\n### PyTorch Equivalents\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.tensor([1,2,3])` | `np.array([1,2,3])` |\n| `torch.zeros(3, 4)` | `np.zeros((3, 4))` |\n| `torch.randn(2, 3)` | `np.random.randn(2, 3)` |\n| `torch.arange(10)` | `np.arange(10)` |\n| `torch.linspace(0, 1, 5)` | `np.linspace(0, 1, 5)` |\n\n### Data Types\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.float32` | `np.float32` |\n| `torch.int64` | `np.int64` |\n\n### Task\nCreate tensors matching PyTorch patterns.\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- `'from_list'`: Array from [1,2,3,4] with float32 dtype\n- `'zeros'`: Zero array of shape (3, 4)\n- `'ones'`: Ones array of shape (2, 3)\n- `'randn'`: Random normal array of shape (2, 3)\n- `'arange'`: Array from 0 to 9\n- `'linspace'`: 5 evenly spaced values from 0 to 1\n- `'eye'`: 4x4 identity matrix\n- `'full'`: (2, 3) array filled with 7.0\n    ",examples:[{input:"create_tensors()",output:"{'zeros': shape (3,4), 'ones': shape (2,3), 'randn': shape (2,3), 'arange': [0..9], ...}",explanation:"Dictionary containing tensors created with various methods"}],starterCode:`import numpy as np

def create_tensors() -> dict:
    """
    Create tensors using various methods.

    Returns:
        Dictionary with named tensors
    """
    np.random.seed(42)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Zeros shape",input:'create_tensors()["zeros"].shape',expected:"(3, 4)",hidden:!1},{id:"2",description:"Random tensor shape",input:'create_tensors()["randn"].shape',expected:"(2, 3)",hidden:!1}],hints:["np.zeros((3, 4)) for 3x4 zeros","np.random.randn(2, 3) for random normal","Specify dtype with astype() or dtype parameter"],solution:`import numpy as np

def create_tensors() -> dict:
    np.random.seed(42)

    return {
        'from_list': np.array([1, 2, 3, 4], dtype=np.float32),
        'zeros': np.zeros((3, 4)),
        'ones': np.ones((2, 3)),
        'randn': np.random.randn(2, 3),
        'arange': np.arange(10),
        'linspace': np.linspace(0, 1, 5),
        'eye': np.eye(4),
        'full': np.full((2, 3), 7.0),
    }
`},{id:"tensor-operations",title:"Tensor Operations",section:"pytorch-basics",difficulty:"easy",description:"\n## Basic Tensor Operations\n\nCommon tensor operations used in neural networks.\n\n### Arithmetic Operations\n- Element-wise: +, -, *, /\n- Matrix multiplication: @ or np.matmul\n- Power: ** or np.power\n\n### Reduction Operations\n- sum, mean, std, var\n- min, max, argmin, argmax\n\n### Shape Operations\n- reshape, view (reshape in NumPy)\n- squeeze, unsqueeze (np.squeeze, np.expand_dims)\n- permute, transpose\n\n### Task\nImplement common tensor operations.\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- **Arithmetic**: `'add'`, `'sub'`, `'mul'`, `'div'`, `'pow'`\n- **Reductions**: `'sum_all'`, `'sum_axis0'`, `'sum_axis1'`, `'mean'`, `'std'`, `'max'`, `'argmax'`\n- **Shape ops**: `'reshape'` (to 3,2), `'flatten'`, `'unsqueeze'` (add dim 0), `'squeeze'`, `'transpose'`\n    ",examples:[{input:"x=[[1,2,3],[4,5,6]], y=[[1,1,1],[2,2,2]]",output:"{'add': [[2,3,4],[6,7,8]], 'mean': 3.5, 'reshape': [[1,2],[3,4],[5,6]], ...}",explanation:"Arithmetic, reduction, and shape operations on tensors"}],starterCode:`import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Perform common tensor operations.

    Args:
        x: First tensor (2, 3)
        y: Second tensor (2, 3)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Addition",input:'bool(np.array_equal(tensor_ops(np.array([[1,2,3],[4,5,6]]), np.array([[1,1,1],[2,2,2]]))["add"], np.array([[2,3,4],[6,7,8]])))',expected:"True",hidden:!1},{id:"2",description:"Mean computation",input:'tensor_ops(np.array([[1,2,3],[4,5,6]]), np.array([[1,1,1],[2,2,2]]))["mean"]',expected:"3.5",hidden:!1}],hints:["Element-wise ops work directly: x + y, x * y","Use axis parameter for reductions along specific dims","np.expand_dims(x, axis=0) adds dimension"],solution:`import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        # Arithmetic
        'add': x + y,
        'sub': x - y,
        'mul': x * y,
        'div': x / (y + 1e-8),
        'pow': x ** 2,

        # Reductions
        'sum_all': x.sum(),
        'sum_axis0': x.sum(axis=0),
        'sum_axis1': x.sum(axis=1),
        'mean': x.mean(),
        'std': x.std(),
        'max': x.max(),
        'argmax': x.argmax(),

        # Shape ops
        'reshape': x.reshape(3, 2),
        'flatten': x.flatten(),
        'unsqueeze': np.expand_dims(x, axis=0),  # (1, 2, 3)
        'squeeze': np.squeeze(np.expand_dims(x, 0)),
        'transpose': x.T,
    }
`},{id:"autograd-concepts",title:"Autograd Concepts (Manual)",section:"pytorch-basics",difficulty:"medium",description:`
## Autograd Concepts

Understand automatic differentiation by implementing it manually.

### PyTorch Autograd
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
\`\`\`

### Manual Gradient Computation
For y = x² + 3x + 1:
- dy/dx = 2x + 3

### Task
Implement forward pass and gradient computation for simple functions.
    `,examples:[{input:"compute_gradients(x=2.0)",output:"{'y': 11.0, 'grad': 7.0}",explanation:"Forward and backward pass"}],starterCode:`import numpy as np

def compute_gradients(x: float) -> dict:
    """
    Compute forward and backward pass for y = x^2 + 3x + 1.

    Args:
        x: Input value

    Returns:
        Dictionary with 'y' (forward result) and 'grad' (dy/dx)
    """
    # Your code here
    pass


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    """
    Compute gradients for a linear layer: Y = X @ W + b

    Args:
        X: Input (batch, in_features)
        W: Weights (in_features, out_features)
        b: Bias (out_features,)
        grad_output: Gradient from next layer (batch, out_features)

    Returns:
        Dictionary with gradients for X, W, and b
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple gradient",input:'compute_gradients(2.0)["grad"]',expected:"7.0",hidden:!1},{id:"2",description:"Forward pass result",input:'compute_gradients(2.0)["y"]',expected:"11.0",hidden:!1}],hints:["For y = x² + 3x + 1, dy/dx = 2x + 3","For Y = X @ W + b: dL/dW = X.T @ grad_output","dL/dX = grad_output @ W.T","dL/db = sum(grad_output, axis=0)"],solution:`import numpy as np

def compute_gradients(x: float) -> dict:
    # Forward pass
    y = x ** 2 + 3 * x + 1

    # Backward pass (analytical gradient)
    grad = 2 * x + 3

    return {'y': y, 'grad': grad}


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    # Forward: Y = X @ W + b

    # Backward:
    # dL/dW = X.T @ grad_output
    grad_W = X.T @ grad_output

    # dL/db = sum of grad_output over batch
    grad_b = grad_output.sum(axis=0)

    # dL/dX = grad_output @ W.T
    grad_X = grad_output @ W.T

    return {
        'grad_W': grad_W,
        'grad_b': grad_b,
        'grad_X': grad_X
    }
`},{id:"nn-modules",title:"Neural Network Modules",section:"pytorch-basics",difficulty:"medium",description:`
## Neural Network Modules

Implement PyTorch-style nn.Module patterns in NumPy.

### Module Pattern
\`\`\`python
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]
\`\`\`

### Task
Implement Linear and ReLU modules with forward method.
    `,examples:[{input:"Linear(10, 5).forward(x)",output:"Output of shape (batch, 5)",explanation:"Linear transformation"}],starterCode:`import numpy as np

class Linear:
    """Linear layer: Y = X @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        # Initialize weights with small random values
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass

    def parameters(self):
        # Return list of parameters
        pass


class ReLU:
    """ReLU activation: max(0, x)"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


class Sequential:
    """Container for sequential layers"""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    """Create a 2-layer MLP: Linear -> ReLU -> Linear"""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Linear output shape",input:"Linear(10, 5).forward(np.random.randn(4, 10)).shape",expected:"(4, 5)",hidden:!1},{id:"2",description:"Sequential forward",input:"create_mlp(10, 20, 5).forward(np.random.randn(4, 10)).shape",expected:"(4, 5)",hidden:!1}],hints:["Initialize W with np.random.randn() * 0.01","ReLU: np.maximum(0, x)","Sequential applies layers in order"],solution:`import numpy as np

class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    return Sequential(
        Linear(input_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, output_dim)
    )
`},{id:"loss-functions",title:"Loss Functions",section:"pytorch-basics",difficulty:"medium",description:`
## Common Loss Functions

Implement loss functions used in deep learning.

### Cross-Entropy Loss
\`\`\`
L = -sum(y * log(softmax(logits)))
\`\`\`

### MSE Loss
\`\`\`
L = mean((y_pred - y_true)²)
\`\`\`

### Binary Cross-Entropy
\`\`\`
L = -mean(y * log(p) + (1-y) * log(1-p))
\`\`\`

### Task
Implement these loss functions with numerical stability.
    `,examples:[{input:"cross_entropy(logits, targets)",output:"Scalar loss value",explanation:"Cross-entropy for classification"}],starterCode:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    # Your code here
    pass


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss for classification.

    Args:
        logits: Raw scores (batch, num_classes)
        targets: Integer class labels (batch,)

    Returns:
        Scalar loss
    """
    # Your code here
    pass


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error loss."""
    # Your code here
    pass


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary cross-entropy with numerical stability."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Cross-entropy computation",input:"round(cross_entropy_loss(np.array([[2.0, 1.0, 0.1]]), np.array([0])), 4)",expected:"0.417",hidden:!1},{id:"2",description:"MSE computation",input:"mse_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))",expected:"0.333333",hidden:!1}],hints:["Softmax: exp(x - max(x)) / sum(exp(x - max(x)))","Use np.clip for numerical stability in log","Index logits with targets for cross-entropy"],solution:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    batch_size = logits.shape[0]
    probs = softmax(logits)

    # Clip for numerical stability
    probs = np.clip(probs, 1e-15, 1 - 1e-15)

    # Select probability of correct class for each sample
    correct_probs = probs[np.arange(batch_size), targets]

    # Negative log likelihood
    loss = -np.mean(np.log(correct_probs))
    return loss


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
`}],T2=[{id:"e2e-mlp",title:"E2E: 2-Layer MLP with Backprop",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End 2-Layer MLP

Implement a complete 2-layer MLP with forward pass, backward pass, and training loop.

### Architecture
\`\`\`
Input (batch, input_dim)
  ↓
Linear(input_dim, hidden_dim) + ReLU
  ↓
Linear(hidden_dim, output_dim)
  ↓
Softmax + Cross-Entropy Loss
\`\`\`

### Forward Pass
\`\`\`python
h = relu(X @ W1 + b1)
logits = h @ W2 + b2
loss = cross_entropy(softmax(logits), y)
\`\`\`

### Backward Pass (Chain Rule)
\`\`\`python
# Output layer
d_logits = softmax(logits) - y_onehot  # (batch, output_dim)
d_W2 = h.T @ d_logits
d_b2 = d_logits.sum(axis=0)

# Hidden layer
d_h = d_logits @ W2.T
d_relu = d_h * (h > 0)  # ReLU derivative
d_W1 = X.T @ d_relu
d_b1 = d_relu.sum(axis=0)
\`\`\`

### Task
Implement forward, backward, and training step.
    `,examples:[{input:"MLP(784, 128, 10) on MNIST-like data",output:"Trained model with decreasing loss",explanation:"Full training loop"}],starterCode:`import numpy as np

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize 2-layer MLP with random weights."""
        np.random.seed(42)
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        # Cache for backward pass
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            X: Input (batch, input_dim)

        Returns:
            logits: Output (batch, output_dim)
        """
        # Your code here
        pass

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model output (batch, output_dim)
            y: True labels (batch,) as integers

        Returns:
            loss: Scalar loss value
        """
        # Your code here
        pass

    def backward(self, y: np.ndarray) -> dict:
        """
        Backward pass.

        Args:
            y: True labels (batch,) as integers

        Returns:
            gradients: Dictionary with dW1, db1, dW2, db2
        """
        # Your code here
        pass

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """
        Single training step: forward, loss, backward, update.

        Returns:
            loss: Loss value before update
        """
        # Your code here
        pass


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    """Train MLP and return loss history."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Forward pass shape",input:"MLP(784, 128, 10).forward(np.random.randn(4, 784)).shape",expected:"(4, 10)",hidden:!1},{id:"2",description:"Loss decreases",input:"(lambda: (m := MLP(10, 8, 3), X := np.random.randn(16, 10), y := np.array([0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0]), losses := [m.train_step(X, y) for _ in range(50)], losses[-1] < losses[0])[-1])()",expected:"True",hidden:!1},{id:"3",description:"Gradients computed",input:'(lambda: (m := MLP(10, 8, 3), X := np.random.randn(4, 10), y := np.array([0,1,2,0]), _ := m.forward(X), _ := m.compute_loss(m.cache["logits"], y), g := m.backward(y), all(k in g for k in ["dW1", "db1", "dW2", "db2"]))[-1])()',expected:"True",hidden:!0}],hints:["Cache X, h (hidden activations), and logits in forward pass","d_logits = probs - y_onehot (softmax gradient combined with cross-entropy)","ReLU gradient: 1 if x > 0, else 0","Update: W = W - lr * dW"],solution:`import numpy as np

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Layer 1: Linear + ReLU
        z1 = X @ self.W1 + self.b1
        h = self.relu(z1)

        # Layer 2: Linear
        logits = h @ self.W2 + self.b2

        # Cache for backward
        self.cache = {'X': X, 'z1': z1, 'h': h, 'logits': logits}

        return logits

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        probs = self.softmax(logits)
        batch_size = logits.shape[0]
        correct_probs = probs[np.arange(batch_size), y]
        loss = -np.mean(np.log(np.clip(correct_probs, 1e-15, 1)))
        self.cache['probs'] = probs
        return loss

    def backward(self, y: np.ndarray) -> dict:
        X = self.cache['X']
        h = self.cache['h']
        z1 = self.cache['z1']
        probs = self.cache['probs']
        batch_size = X.shape[0]

        # One-hot encode targets
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(batch_size), y] = 1

        # Output layer gradients
        d_logits = (probs - y_onehot) / batch_size
        dW2 = h.T @ d_logits
        db2 = d_logits.sum(axis=0)

        # Hidden layer gradients
        d_h = d_logits @ self.W2.T
        d_relu = d_h * (z1 > 0)  # ReLU derivative
        dW1 = X.T @ d_relu
        db1 = d_relu.sum(axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        logits = self.forward(X)
        loss = self.compute_loss(logits, y)
        grads = self.backward(y)

        # Update weights
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']

        return loss


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    model = MLP(input_dim, 64, output_dim)

    losses = []
    for _ in range(epochs):
        loss = model.train_step(X_train, y_train, lr)
        losses.append(loss)

    return model, losses
`},{id:"e2e-transformer",title:"E2E: Transformer Encoder",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Transformer Encoder

Implement a complete Transformer encoder block.

### Architecture
\`\`\`
Input Embeddings + Positional Encoding
  ↓
Multi-Head Self-Attention
  ↓
Add & LayerNorm (Residual)
  ↓
Feed-Forward Network (FFN)
  ↓
Add & LayerNorm (Residual)
  ↓
Output
\`\`\`

### Components
1. **Positional Encoding**: Sinusoidal position embeddings
2. **Multi-Head Attention**: Parallel attention heads
3. **FFN**: Two linear layers with ReLU
4. **Layer Normalization**: Normalize across features
5. **Residual Connections**: Add input to output

### Task
Implement all components and combine into encoder block.
    `,examples:[{input:"TransformerEncoder(d_model=64, nhead=8, dim_ff=256)",output:"Encoded sequence (batch, seq_len, d_model)",explanation:"Complete encoder forward pass"}],starterCode:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    # Your code here
    pass


def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding."""
    # Your code here
    pass


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    # Your code here
    pass


def multi_head_attention(Q, K, V, num_heads):
    """Multi-head attention."""
    # Your code here
    pass


def feed_forward(x, W1, b1, W2, b2):
    """Position-wise feed-forward network: Linear -> ReLU -> Linear"""
    # Your code here
    pass


class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """
        Initialize transformer encoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        np.random.seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Initialize weights
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through encoder block.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Your code here
        pass


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    """Stack multiple encoder blocks."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"transformer_encoder(np.random.randn(2, 10, 64), num_layers=2, d_model=64, num_heads=8, d_ff=256).shape",expected:"(2, 10, 64)",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"(lambda: (Q := np.random.randn(2, 5, 64), K := np.random.randn(2, 5, 64), V := np.random.randn(2, 5, 64), out_attn := scaled_dot_product_attention(Q, K, V), bool(np.allclose(out_attn[1].sum(axis=-1), 1.0)))[-1])()",expected:"True",hidden:!1}],hints:["Attention: softmax(QK^T / sqrt(d_k)) @ V","Layer norm: (x - mean) / sqrt(var + eps) * gamma + beta","Residual: output = LayerNorm(x + Sublayer(x))","FFN: max(0, xW1 + b1)W2 + b2"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attention = softmax(scores, axis=-1)
    return attention @ V, attention


def multi_head_attention(Q, K, V, num_heads):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    d_k = Q.shape[-1]
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(d_k)
    attention = softmax(scores, axis=-1)
    output = np.einsum('bhqk,bhkd->bhqd', attention, V)

    output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return output


def feed_forward(x, W1, b1, W2, b2):
    hidden = np.maximum(0, x @ W1 + b1)
    return hidden @ W2 + b2


class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        np.random.seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

        # FFN parameters
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Multi-head self-attention + residual + layer norm
        attn_output = multi_head_attention(x, x, x, self.num_heads)
        x = layer_norm(x + attn_output, self.ln1_gamma, self.ln1_beta)

        # Feed-forward + residual + layer norm
        ff_output = feed_forward(x, self.W1, self.b1, self.W2, self.b2)
        x = layer_norm(x + ff_output, self.ln2_gamma, self.ln2_beta)

        return x


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    batch, seq_len, _ = x.shape

    # Add positional encoding
    PE = positional_encoding(seq_len, d_model)
    x = x + PE

    # Stack encoder blocks
    for _ in range(num_layers):
        block = TransformerEncoderBlock(d_model, num_heads, d_ff)
        x = block.forward(x)

    return x
`},{id:"e2e-vae",title:"E2E: Variational Autoencoder",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Variational Autoencoder

Implement a complete VAE with encoder, decoder, reparameterization, and ELBO loss.

### Architecture
\`\`\`
Encoder: x → h → (μ, log_σ²)
         ↓
Reparameterization: z = μ + σ * ε, ε ~ N(0,1)
         ↓
Decoder: z → h → x_reconstructed
\`\`\`

### Loss (ELBO)
\`\`\`
L = Reconstruction Loss + KL Divergence
L = ||x - x_hat||² + KL(q(z|x) || p(z))
KL = -0.5 * sum(1 + log_var - μ² - exp(log_var))
\`\`\`

### Task
Implement encoder, decoder, reparameterization trick, and loss computation.
    `,examples:[{input:"VAE(input_dim=784, latent_dim=20)",output:"Reconstructed images + latent samples",explanation:"Full VAE forward pass"}],starterCode:`import numpy as np

class VAE:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Initialize VAE.

        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        np.random.seed(42)

        # Encoder weights
        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc = np.zeros(hidden_dim)

        # Latent space (mean and log variance)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

        # Decoder weights
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec2 = np.zeros(input_dim)

        self.latent_dim = latent_dim

    def encode(self, x: np.ndarray) -> tuple:
        """
        Encode input to latent distribution parameters.

        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        # Your code here
        pass

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        # Your code here
        pass

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector to reconstruction.
        """
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> dict:
        """
        Full forward pass.

        Returns:
            Dictionary with mu, log_var, z, x_reconstructed
        """
        # Your code here
        pass

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     mu: np.ndarray, log_var: np.ndarray) -> dict:
        """
        Compute VAE loss (ELBO).

        Returns:
            Dictionary with total_loss, recon_loss, kl_loss
        """
        # Your code here
        pass

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generate samples by decoding random latent vectors.
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Reconstruction shape",input:'VAE(784, 256, 20).forward(np.random.randn(4, 784))["x_recon"].shape',expected:"(4, 784)",hidden:!1},{id:"2",description:"KL divergence non-negative",input:'(lambda: (vae := VAE(784, 256, 20), x := np.random.randn(4, 784), out := vae.forward(x), loss := vae.compute_loss(x, out["x_recon"], out["mu"], out["log_var"]), bool(loss["kl_loss"] >= 0))[-1])()',expected:"True",hidden:!1}],hints:["Encoder: h = relu(x @ W + b), then mu = h @ W_mu, logvar = h @ W_logvar","Reparameterize: std = exp(0.5 * log_var), z = mu + std * epsilon","KL = -0.5 * mean(1 + log_var - mu² - exp(log_var))","Reconstruction loss: MSE or BCE"],solution:`import numpy as np

class VAE:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        np.random.seed(42)

        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec2 = np.zeros(input_dim)
        self.latent_dim = latent_dim

    def encode(self, x: np.ndarray) -> tuple:
        h = np.maximum(0, x @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        return mu, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = np.maximum(0, z @ self.W_dec1 + self.b_dec1)
        x_recon = 1 / (1 + np.exp(-(h @ self.W_dec2 + self.b_dec2)))  # Sigmoid
        return x_recon

    def forward(self, x: np.ndarray) -> dict:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return {'mu': mu, 'log_var': log_var, 'z': z, 'x_recon': x_recon}

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     mu: np.ndarray, log_var: np.ndarray) -> dict:
        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - x_recon) ** 2)

        # KL divergence
        kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

        total_loss = recon_loss + kl_loss

        return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

    def sample(self, num_samples: int) -> np.ndarray:
        z = np.random.randn(num_samples, self.latent_dim)
        return self.decode(z)
`},{id:"e2e-vqvae",title:"E2E: Vector Quantized VAE",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End VQ-VAE

Implement a complete VQ-VAE (Vector Quantized Variational Autoencoder) with encoder, vector quantization, and decoder.

### Architecture
\`\`\`
Encoder: x → z_e (continuous latents)
           ↓
Vector Quantization: z_e → z_q (discrete latents from codebook)
           ↓
Decoder: z_q → x_reconstructed
\`\`\`

### Key Difference from VAE
- VAE: Continuous latent space with KL regularization
- VQ-VAE: Discrete latent space using learned codebook

### Vector Quantization
1. Encoder outputs continuous vectors z_e
2. Find nearest codebook entry for each spatial position
3. Replace z_e with quantized z_q from codebook
4. Use straight-through estimator for gradients

### Loss Function
\`\`\`
L = L_reconstruction + L_codebook + β * L_commitment

L_reconstruction = ||x - x_hat||²
L_codebook = ||sg[z_e] - e||²  (moves codebook toward encoder output)
L_commitment = ||z_e - sg[e]||²  (commits encoder to codebook)
\`\`\`
Where sg[] is stop-gradient.

### Codebook EMA Update (Alternative)
Instead of gradient updates, codebook can be updated with exponential moving average:
\`\`\`
N_i = γ * N_i + (1 - γ) * n_i     (count of assignments)
m_i = γ * m_i + (1 - γ) * sum(z_e assigned to i)
e_i = m_i / N_i
\`\`\`

### Task
Implement encoder, vector quantization layer, decoder, and all loss components.
    `,examples:[{input:"VQVAE(input_dim=784, num_embeddings=512, embedding_dim=64)",output:"Reconstructed images + discrete codes",explanation:"Full VQ-VAE forward pass"}],starterCode:`import numpy as np

class VectorQuantizer:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Vector Quantization layer.

        Args:
            num_embeddings: Size of codebook (K)
            embedding_dim: Dimension of each embedding
        """
        np.random.seed(42)
        # Initialize codebook with small random values
        self.embedding = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: np.ndarray) -> tuple:
        """
        Quantize encoder output to nearest codebook entries.

        Args:
            z_e: Encoder output (batch, height, width, embedding_dim)

        Returns:
            z_q: Quantized vectors (same shape as z_e)
            indices: Codebook indices (batch, height, width)
            distances: Distances to nearest embeddings
        """
        # Your code here
        pass

    def compute_loss(self, z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        """
        Compute VQ losses.

        Args:
            z_e: Encoder output
            z_q: Quantized vectors
            beta: Commitment loss weight

        Returns:
            Dictionary with codebook_loss and commitment_loss
        """
        # Your code here
        pass


class VQVAE:
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        """
        Initialize VQ-VAE.

        Args:
            input_channels: Number of input channels
            hidden_dim: Hidden layer dimension
            num_embeddings: Codebook size
            embedding_dim: Embedding dimension
        """
        np.random.seed(42)

        # Encoder: conv layers to get spatial feature maps
        self.enc_conv1_w = np.random.randn(hidden_dim, input_channels, 4, 4) * 0.1
        self.enc_conv1_b = np.zeros(hidden_dim)
        self.enc_conv2_w = np.random.randn(embedding_dim, hidden_dim, 4, 4) * 0.1
        self.enc_conv2_b = np.zeros(embedding_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder: transposed conv to reconstruct
        self.dec_conv1_w = np.random.randn(hidden_dim, embedding_dim, 4, 4) * 0.1
        self.dec_conv1_b = np.zeros(hidden_dim)
        self.dec_conv2_w = np.random.randn(input_channels, hidden_dim, 4, 4) * 0.1
        self.dec_conv2_b = np.zeros(input_channels)

        self.embedding_dim = embedding_dim

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to continuous latent representation.

        Args:
            x: Input (batch, channels, height, width)

        Returns:
            z_e: Encoder output (batch, height', width', embedding_dim)
        """
        # Your code here (simplified: use strided conv or just reshape for demo)
        pass

    def decode(self, z_q: np.ndarray) -> np.ndarray:
        """
        Decode quantized latents to reconstruction.

        Args:
            z_q: Quantized vectors (batch, height, width, embedding_dim)

        Returns:
            x_recon: Reconstructed input
        """
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> dict:
        """
        Full forward pass.

        Returns:
            Dictionary with z_e, z_q, indices, x_recon
        """
        # Your code here
        pass

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        """
        Compute total VQ-VAE loss.

        Returns:
            Dictionary with total_loss, recon_loss, vq_loss, commitment_loss
        """
        # Your code here
        pass

    def get_codebook_usage(self, indices: np.ndarray) -> np.ndarray:
        """
        Compute codebook usage statistics.

        Returns:
            usage: Count of each codebook entry used
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Quantization shape",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), z_q, idx, dist := vq.quantize(z_e), z_q.shape == z_e.shape)[-1])()",expected:"True",hidden:!1},{id:"2",description:"Indices valid range",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), z_q, idx, dist := vq.quantize(z_e), bool(np.all(idx >= 0) and np.all(idx < 512)))[-1])()",expected:"True",hidden:!1},{id:"3",description:"Quantized from codebook",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), z_q, idx, dist := vq.quantize(z_e), bool(np.allclose(z_q[0, 0, 0], vq.embedding[idx[0, 0, 0]])))[-1])()",expected:"True",hidden:!0},{id:"4",description:"Loss values correct",input:'(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), z_q, idx, dist := vq.quantize(z_e), losses := vq.compute_loss(z_e, z_q), bool(losses["codebook_loss"] >= 0 and losses["commitment_loss"] >= 0))[-1])()',expected:"True",hidden:!0}],hints:["For quantization: flatten spatial dims, compute distances, find argmin","Distance: ||z - e||² = ||z||² + ||e||² - 2*z·e","Straight-through: z_q = z_e + stop_grad(z_q - z_e)","Codebook loss uses z_e detached, commitment uses z_q detached","For simplified encoder/decoder, use reshape operations"],solution:`import numpy as np

class VectorQuantizer:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        np.random.seed(42)
        self.embedding = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: np.ndarray) -> tuple:
        # z_e: (batch, H, W, D)
        batch, H, W, D = z_e.shape

        # Flatten spatial dimensions
        z_flat = z_e.reshape(-1, D)  # (N, D) where N = batch * H * W

        # Compute distances: ||z - e||² = ||z||² + ||e||² - 2*z·e
        z_sq = np.sum(z_flat ** 2, axis=1, keepdims=True)  # (N, 1)
        e_sq = np.sum(self.embedding ** 2, axis=1)          # (K,)
        cross = z_flat @ self.embedding.T                    # (N, K)
        distances = z_sq + e_sq - 2 * cross                  # (N, K)

        # Find nearest embedding
        indices_flat = np.argmin(distances, axis=1)          # (N,)

        # Get quantized vectors
        z_q_flat = self.embedding[indices_flat]              # (N, D)

        # Reshape back
        z_q = z_q_flat.reshape(batch, H, W, D)
        indices = indices_flat.reshape(batch, H, W)
        min_distances = np.min(distances, axis=1).reshape(batch, H, W)

        return z_q, indices, min_distances

    def compute_loss(self, z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        # Codebook loss: ||sg[z_e] - z_q||²
        # In practice, this updates the codebook to move toward encoder output
        codebook_loss = np.mean((z_e - z_q) ** 2)

        # Commitment loss: ||z_e - sg[z_q]||²
        # This commits the encoder output to stay close to codebook
        commitment_loss = beta * np.mean((z_e - z_q) ** 2)

        return {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }


class VQVAE:
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        np.random.seed(42)

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Simplified: use linear projection instead of conv for demo
        # Encoder projects input to embedding space
        self.enc_w1 = np.random.randn(input_channels * 16, hidden_dim) * 0.1
        self.enc_b1 = np.zeros(hidden_dim)
        self.enc_w2 = np.random.randn(hidden_dim, embedding_dim) * 0.1
        self.enc_b2 = np.zeros(embedding_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder projects back
        self.dec_w1 = np.random.randn(embedding_dim, hidden_dim) * 0.1
        self.dec_b1 = np.zeros(hidden_dim)
        self.dec_w2 = np.random.randn(hidden_dim, input_channels * 16) * 0.1
        self.dec_b2 = np.zeros(input_channels * 16)

    def encode(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, channels, H, W) - assume H=W=4 for simplicity
        batch = x.shape[0]

        # Flatten spatial dimensions
        x_flat = x.reshape(batch, -1)  # (batch, channels * H * W)

        # Encode
        h = np.maximum(0, x_flat @ self.enc_w1 + self.enc_b1)  # ReLU
        z_e = x_flat @ self.enc_w1[:, :self.embedding_dim] + self.enc_b1[:self.embedding_dim]

        # Reshape to spatial format (batch, 2, 2, embedding_dim)
        z_e = z_e.reshape(batch, 1, 1, self.embedding_dim)

        return z_e

    def decode(self, z_q: np.ndarray) -> np.ndarray:
        batch = z_q.shape[0]

        # Flatten
        z_flat = z_q.reshape(batch, -1)  # (batch, embedding_dim)

        # Decode
        h = np.maximum(0, z_flat @ self.dec_w1 + self.dec_b1)  # ReLU
        x_recon = h @ self.dec_w2 + self.dec_b2

        # Reshape to image format
        x_recon = x_recon.reshape(batch, self.input_channels, 4, 4)

        return x_recon

    def forward(self, x: np.ndarray) -> dict:
        # Encode
        z_e = self.encode(x)

        # Quantize
        z_q, indices, distances = self.vq.quantize(z_e)

        # Straight-through estimator: gradient flows through z_q to z_e
        # z_q_st = z_e + (z_q - z_e).detach()
        # For forward pass, we just use z_q

        # Decode
        x_recon = self.decode(z_q)

        return {
            'z_e': z_e,
            'z_q': z_q,
            'indices': indices,
            'x_recon': x_recon
        }

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        # Reconstruction loss
        recon_loss = np.mean((x - x_recon) ** 2)

        # VQ losses
        vq_losses = self.vq.compute_loss(z_e, z_q, beta)

        # Total loss
        total_loss = recon_loss + vq_losses['codebook_loss'] + vq_losses['commitment_loss']

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'codebook_loss': vq_losses['codebook_loss'],
            'commitment_loss': vq_losses['commitment_loss']
        }

    def get_codebook_usage(self, indices: np.ndarray) -> np.ndarray:
        usage = np.bincount(indices.flatten(), minlength=self.vq.num_embeddings)
        return usage
`},{id:"e2e-diffusion",title:"E2E: Diffusion Model",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Diffusion Model

Implement core components of a diffusion model (DDPM-style).

### Forward Process (Adding Noise)
\`\`\`
x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
\`\`\`

### Noise Schedule
\`\`\`
β_t: Linear from β_start to β_end
α_t = 1 - β_t
α_bar_t = cumulative product of α_t
\`\`\`

### Training Objective
\`\`\`
L = ||ε - ε_θ(x_t, t)||²
\`\`\`

Model learns to predict the noise added at timestep t.

### Reverse Process (Sampling)
\`\`\`
x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-α_bar_t)) * ε_θ(x_t, t)) + σ_t * z
\`\`\`

### Task
Implement noise schedule, forward process, and reverse sampling.
    `,examples:[{input:"DiffusionModel(T=1000)",output:"Denoised samples",explanation:"Forward and reverse diffusion"}],starterCode:`import numpy as np

class DiffusionModel:
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Initialize diffusion model.

        Args:
            T: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.T = T

        # Compute noise schedule
        # Your code here
        pass

    def get_noise_schedule(self) -> dict:
        """Return computed noise schedule parameters."""
        # Your code here
        pass

    def forward_diffusion(self, x_0: np.ndarray, t: int) -> tuple:
        """
        Add noise to data at timestep t.

        Args:
            x_0: Original data (batch, ...)
            t: Timestep

        Returns:
            x_t: Noisy data
            noise: The noise that was added
        """
        np.random.seed(42)
        # Your code here
        pass

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        """
        Single reverse diffusion step.

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            predicted_noise: Noise predicted by model

        Returns:
            x_{t-1}: Less noisy data
        """
        # Your code here
        pass

    def compute_loss(self, true_noise: np.ndarray, predicted_noise: np.ndarray) -> float:
        """Compute MSE loss between true and predicted noise."""
        # Your code here
        pass

    def sample(self, shape: tuple, noise_predictor) -> np.ndarray:
        """
        Generate samples using reverse diffusion.

        Args:
            shape: Shape of samples to generate
            noise_predictor: Function that predicts noise given (x_t, t)

        Returns:
            Generated samples
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Alpha bar decreases",input:'(lambda: (dm := DiffusionModel(1000), sched := dm.get_noise_schedule(), bool(sched["alpha_bars"][-1] < sched["alpha_bars"][0]))[-1])()',expected:"True",hidden:!1},{id:"2",description:"Forward noise shape",input:"(lambda: (dm := DiffusionModel(1000), x_0 := np.random.randn(4, 32), x_t, noise := dm.forward_diffusion(x_0, 500), x_t.shape == x_0.shape)[-1])()",expected:"True",hidden:!1}],hints:["betas = np.linspace(beta_start, beta_end, T)","alpha_bars = np.cumprod(1 - betas)","x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise","Reverse: use mean and add small noise (except at t=0)"],solution:`import numpy as np

class DiffusionModel:
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.T = T

        # Linear noise schedule
        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        # For reverse process
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1 - self.alpha_bars)
        self.sqrt_alphas = np.sqrt(self.alphas)

    def get_noise_schedule(self) -> dict:
        return {
            'betas': self.betas,
            'alphas': self.alphas,
            'alpha_bars': self.alpha_bars
        }

    def forward_diffusion(self, x_0: np.ndarray, t: int) -> tuple:
        np.random.seed(42)
        noise = np.random.randn(*x_0.shape)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        # Mean of reverse distribution
        coef1 = 1 / np.sqrt(alpha)
        coef2 = beta / np.sqrt(1 - alpha_bar)
        mean = coef1 * (x_t - coef2 * predicted_noise)

        if t > 0:
            # Add noise (except at final step)
            sigma = np.sqrt(beta)
            noise = np.random.randn(*x_t.shape)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean

        return x_prev

    def compute_loss(self, true_noise: np.ndarray, predicted_noise: np.ndarray) -> float:
        return np.mean((true_noise - predicted_noise) ** 2)

    def sample(self, shape: tuple, noise_predictor) -> np.ndarray:
        # Start from pure noise
        x = np.random.randn(*shape)

        # Reverse diffusion
        for t in reversed(range(self.T)):
            predicted_noise = noise_predictor(x, t)
            x = self.reverse_step(x, t, predicted_noise)

        return x
`},{id:"e2e-cnn",title:"E2E: Convolutional Neural Network",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End CNN

Implement a complete CNN for image classification.

### Architecture
\`\`\`
Input (batch, channels, height, width)
  ↓
Conv2D(in_channels, 32, kernel=3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Conv2D(32, 64, kernel=3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Flatten
  ↓
Linear(flattened_size, 128) + ReLU
  ↓
Linear(128, num_classes)
  ↓
Softmax
\`\`\`

### Components
- 2D Convolution (no padding for simplicity)
- Max Pooling
- Flatten
- Fully Connected layers

### Task
Implement conv2d, maxpool, and combine into CNN.
    `,examples:[{input:"CNN on 28x28 grayscale images",output:"Class probabilities",explanation:"Full CNN forward pass"}],starterCode:`import numpy as np

def conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    2D convolution (no padding, stride=1).

    Args:
        x: Input (batch, in_channels, height, width)
        kernel: Weights (out_channels, in_channels, kH, kW)
        bias: Bias (out_channels,)

    Returns:
        output: (batch, out_channels, out_height, out_width)
    """
    # Your code here
    pass


def maxpool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    2D max pooling.

    Args:
        x: Input (batch, channels, height, width)
        pool_size: Size of pooling window

    Returns:
        output: Pooled tensor
    """
    # Your code here
    pass


def flatten(x: np.ndarray) -> np.ndarray:
    """Flatten all dimensions except batch."""
    # Your code here
    pass


class CNN:
    def __init__(self, input_shape: tuple, num_classes: int):
        """
        Initialize CNN.

        Args:
            input_shape: (channels, height, width)
            num_classes: Number of output classes
        """
        np.random.seed(42)
        in_channels, H, W = input_shape

        # Conv layer 1: in_channels -> 32
        self.conv1_w = np.random.randn(32, in_channels, 3, 3) * 0.1
        self.conv1_b = np.zeros(32)

        # Conv layer 2: 32 -> 64
        self.conv2_w = np.random.randn(64, 32, 3, 3) * 0.1
        self.conv2_b = np.zeros(64)

        # Calculate flattened size after convolutions and pooling
        # Your code to compute this
        pass

        # Fully connected layers
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through CNN.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            logits: (batch, num_classes)
        """
        # Your code here
        pass


def test_cnn():
    """Test CNN with sample data."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Conv output shape",input:"conv2d(np.random.randn(2, 1, 8, 8), np.random.randn(32, 1, 3, 3), np.zeros(32)).shape",expected:"(2, 32, 6, 6)",hidden:!1},{id:"2",description:"Pool output shape",input:"maxpool2d(np.random.randn(2, 32, 8, 8), pool_size=2).shape",expected:"(2, 32, 4, 4)",hidden:!1}],hints:["Conv output size: (input - kernel + 1) for stride=1, no padding","Pool output size: input // pool_size","Use nested loops for convolution (or np.lib.stride_tricks)","Flatten: x.reshape(batch_size, -1)"],solution:`import numpy as np

def conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, in_ch, H, W = x.shape
    out_ch, _, kH, kW = kernel.shape
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((batch, out_ch, out_H, out_W))

    for b in range(batch):
        for oc in range(out_ch):
            for i in range(out_H):
                for j in range(out_W):
                    receptive_field = x[b, :, i:i+kH, j:j+kW]
                    output[b, oc, i, j] = np.sum(receptive_field * kernel[oc]) + bias[oc]

    return output


def maxpool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    batch, channels, H, W = x.shape
    out_H = H // pool_size
    out_W = W // pool_size

    output = np.zeros((batch, channels, out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * pool_size
            w_start = j * pool_size
            output[:, :, i, j] = np.max(
                x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size],
                axis=(2, 3)
            )

    return output


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


class CNN:
    def __init__(self, input_shape: tuple, num_classes: int):
        np.random.seed(42)
        in_channels, H, W = input_shape

        # Conv layers
        self.conv1_w = np.random.randn(32, in_channels, 3, 3) * 0.1
        self.conv1_b = np.zeros(32)
        self.conv2_w = np.random.randn(64, 32, 3, 3) * 0.1
        self.conv2_b = np.zeros(64)

        # Calculate size after convolutions and pooling
        # After conv1 (3x3): H-2, W-2
        # After pool1 (2x2): (H-2)//2, (W-2)//2
        # After conv2 (3x3): (H-2)//2 - 2, (W-2)//2 - 2
        # After pool2 (2x2): ((H-2)//2 - 2)//2, ((W-2)//2 - 2)//2
        h1 = (H - 2) // 2
        w1 = (W - 2) // 2
        h2 = (h1 - 2) // 2
        w2 = (w1 - 2) // 2
        flat_size = 64 * h2 * w2

        # FC layers
        self.fc1_w = np.random.randn(flat_size, 128) * 0.1
        self.fc1_b = np.zeros(128)
        self.fc2_w = np.random.randn(128, num_classes) * 0.1
        self.fc2_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Conv block 1
        x = conv2d(x, self.conv1_w, self.conv1_b)
        x = np.maximum(0, x)  # ReLU
        x = maxpool2d(x)

        # Conv block 2
        x = conv2d(x, self.conv2_w, self.conv2_b)
        x = np.maximum(0, x)  # ReLU
        x = maxpool2d(x)

        # Flatten
        x = flatten(x)

        # FC layers
        x = np.maximum(0, x @ self.fc1_w + self.fc1_b)  # ReLU
        x = x @ self.fc2_w + self.fc2_b  # Logits

        return x


def test_cnn():
    # Test with MNIST-like data
    cnn = CNN(input_shape=(1, 28, 28), num_classes=10)
    x = np.random.randn(4, 1, 28, 28)
    logits = cnn.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    return logits.shape == (4, 10)
`}],Cg=[...h2,...g2,...y2,...x2,...v2,..._2,...w2,...b2,...k2,...S2,...C2,...E2,...z2,...T2];function P2(e){return Cg.find(t=>t.id===e)}function N2(e){return Cg.filter(t=>t.section===e)}function A2(){const{sectionId:e}=Ph(),{getProblemProgress:t,getSectionProgress:n}=na(),r=Mr.find(o=>o.id===e),i=e?N2(e):[];if(!r)return h.jsxs("div",{className:"text-center py-12",children:[h.jsx("h1",{className:"text-2xl font-bold text-gray-900 mb-4",children:"Section Not Found"}),h.jsx(Nt,{to:"/",className:"text-primary-600 hover:text-primary-500",children:"Return to Home"})]});const a=n(r.id,r.problems.length);return h.jsxs("div",{className:"max-w-4xl mx-auto",children:[h.jsx(Qo,{title:r.title,description:`${r.description} Learn ${r.title.toLowerCase()} with ${r.problems.length} hands-on coding problems.`,canonical:`/section/${r.id}`}),h.jsxs("div",{className:"mb-8",children:[h.jsxs("div",{className:"flex items-center gap-4 mb-4",children:[h.jsx("div",{className:"w-14 h-14 bg-white border border-gray-200 rounded-xl flex items-center justify-center text-3xl shadow-sm",children:r.icon}),h.jsxs("div",{children:[h.jsx("h1",{className:"text-2xl font-bold text-gray-900",children:r.title}),h.jsx("p",{className:"text-gray-500",children:r.description})]})]}),h.jsxs("div",{className:"flex items-center gap-4",children:[h.jsx("div",{className:"flex-1 h-2 bg-gray-200 rounded-full overflow-hidden",children:h.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${a}%`}})}),h.jsxs("span",{className:"text-primary-600 font-medium",children:[a,"% complete"]})]})]}),h.jsxs("div",{className:"bg-white rounded-xl p-6 mb-8 border border-gray-200 shadow-sm",children:[h.jsx("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Introduction"}),h.jsx("div",{className:"prose prose-sm max-w-none",children:h.jsx(Sg,{components:{h1:({children:o})=>h.jsx("h1",{className:"text-xl font-bold text-gray-900 mt-0 mb-4",children:o}),h2:({children:o})=>h.jsx("h2",{className:"text-lg font-semibold text-gray-900 mt-6 mb-3",children:o}),h3:({children:o})=>h.jsx("h3",{className:"text-base font-medium text-gray-700 mt-4 mb-2",children:o}),p:({children:o})=>h.jsx("p",{className:"text-gray-600 mb-3 leading-relaxed",children:o}),code:({className:o,children:s})=>(o==null?void 0:o.includes("language-"))?h.jsx("pre",{className:"bg-gray-100 rounded-lg p-4 overflow-x-auto my-3",children:h.jsx("code",{className:"text-sm text-gray-800",children:s})}):h.jsx("code",{className:"bg-gray-100 px-1.5 py-0.5 rounded text-primary-600 text-sm",children:s}),ul:({children:o})=>h.jsx("ul",{className:"list-disc list-inside text-gray-600 space-y-1 mb-3",children:o}),li:({children:o})=>h.jsx("li",{className:"text-gray-600",children:o}),strong:({children:o})=>h.jsx("strong",{className:"text-gray-900 font-semibold",children:o})},children:r.introduction})})]}),h.jsxs("div",{children:[h.jsx("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Problems"}),h.jsx("div",{className:"space-y-3",children:i.map((o,s)=>{const l=t(r.id,o.id);return h.jsxs(Nt,{to:`/problem/${r.id}/${o.id}`,className:"flex items-center gap-4 bg-white rounded-lg p-4 border border-gray-200 hover:border-primary-400 hover:shadow-sm transition-all group",children:[h.jsx("div",{className:`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${l.status==="completed"?"bg-green-100 text-green-600":l.status==="in_progress"?"bg-yellow-100 text-yellow-600":"bg-gray-100 text-gray-500"}`,children:l.status==="completed"?"✓":s+1}),h.jsxs("div",{className:"flex-1",children:[h.jsx("h3",{className:"text-gray-900 font-medium group-hover:text-primary-600 transition-colors",children:o.title}),h.jsxs("div",{className:"flex items-center gap-3 mt-1",children:[h.jsx("span",{className:`text-xs px-2 py-0.5 rounded ${o.difficulty==="easy"?"bg-green-100 text-green-700":o.difficulty==="medium"?"bg-yellow-100 text-yellow-700":"bg-red-100 text-red-700"}`,children:o.difficulty}),h.jsxs("span",{className:"text-gray-400 text-xs",children:[o.testCases.length," test cases"]})]})]}),h.jsx("div",{className:"text-gray-400 group-hover:text-primary-600 transition-colors",children:"→"})]},o.id)})})]})]})}var Eg={exports:{}},L2="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED",j2=L2,R2=j2;function zg(){}function Tg(){}Tg.resetWarningCache=zg;var I2=function(){function e(r,i,a,o,s,l){if(l!==R2){var u=new Error("Calling PropTypes validators directly is not supported by the `prop-types` package. Use PropTypes.checkPropTypes() to call them. Read more at http://fb.me/use-check-prop-types");throw u.name="Invariant Violation",u}}e.isRequired=e;function t(){return e}var n={array:e,bigint:e,bool:e,func:e,number:e,object:e,string:e,symbol:e,any:e,arrayOf:t,element:e,elementType:e,instanceOf:t,node:e,objectOf:t,oneOf:t,oneOfType:t,shape:t,exact:t,checkPropTypes:Tg,resetWarningCache:zg};return n.PropTypes=n,n};Eg.exports=I2();var O2=Eg.exports;const oe=Zn(O2);var pt=typeof window<"u"?window:null,zc=pt===null,$i=zc?void 0:pt.document,kt="addEventListener",St="removeEventListener",Vs="getBoundingClientRect",oi="_a",Ct="_b",Qt="_c",Ta="horizontal",Et=function(){return!1},D2=zc?"calc":["","-webkit-","-moz-","-o-"].filter(function(e){var t=$i.createElement("div");return t.style.cssText="width:"+e+"calc(9px)",!!t.style.length}).shift()+"calc",Pg=function(e){return typeof e=="string"||e instanceof String},Vp=function(e){if(Pg(e)){var t=$i.querySelector(e);if(!t)throw new Error("Selector "+e+" did not match a DOM element");return t}return e},De=function(e,t,n){var r=e[t];return r!==void 0?r:n},Pa=function(e,t,n,r){if(t){if(r==="end")return 0;if(r==="center")return e/2}else if(n){if(r==="start")return 0;if(r==="center")return e/2}return e},M2=function(e,t){var n=$i.createElement("div");return n.className="gutter gutter-"+t,n},F2=function(e,t,n){var r={};return Pg(t)?r[e]=t:r[e]=D2+"("+t+"% - "+n+"px)",r},q2=function(e,t){var n;return n={},n[e]=t+"px",n},Hp=function(e,t){if(t===void 0&&(t={}),zc)return{};var n=e,r,i,a,o,s,l;Array.from&&(n=Array.from(n));var u=Vp(n[0]),d=u.parentNode,c=getComputedStyle?getComputedStyle(d):null,p=c?c.flexDirection:null,f=De(t,"sizes")||n.map(function(){return 100/n.length}),y=De(t,"minSize",100),v=Array.isArray(y)?y:n.map(function(){return y}),k=De(t,"maxSize",1/0),m=Array.isArray(k)?k:n.map(function(){return k}),g=De(t,"expandToMin",!1),x=De(t,"gutterSize",10),C=De(t,"gutterAlign","center"),z=De(t,"snapOffset",30),S=Array.isArray(z)?z:n.map(function(){return z}),T=De(t,"dragInterval",1),N=De(t,"direction",Ta),F=De(t,"cursor",N===Ta?"col-resize":"row-resize"),b=De(t,"gutter",M2),I=De(t,"elementStyle",F2),W=De(t,"gutterStyle",q2);N===Ta?(r="width",i="clientX",a="left",o="right",s="clientWidth"):N==="vertical"&&(r="height",i="clientY",a="top",o="bottom",s="clientHeight");function X(q,L,D,H){var Ee=I(r,L,D,H);Object.keys(Ee).forEach(function(ve){q.style[ve]=Ee[ve]})}function ee(q,L,D){var H=W(r,L,D);Object.keys(H).forEach(function(Ee){q.style[Ee]=H[Ee]})}function A(){return l.map(function(q){return q.size})}function $(q){return"touches"in q?q.touches[0][i]:q[i]}function ne(q){var L=l[this.a],D=l[this.b],H=L.size+D.size;L.size=q/this.size*H,D.size=H-q/this.size*H,X(L.element,L.size,this[Ct],L.i),X(D.element,D.size,this[Qt],D.i)}function O(q){var L,D=l[this.a],H=l[this.b];this.dragging&&(L=$(q)-this.start+(this[Ct]-this.dragOffset),T>1&&(L=Math.round(L/T)*T),L<=D.minSize+D.snapOffset+this[Ct]?L=D.minSize+this[Ct]:L>=this.size-(H.minSize+H.snapOffset+this[Qt])&&(L=this.size-(H.minSize+this[Qt])),L>=D.maxSize-D.snapOffset+this[Ct]?L=D.maxSize+this[Ct]:L<=this.size-(H.maxSize-H.snapOffset+this[Qt])&&(L=this.size-(H.maxSize+this[Qt])),ne.call(this,L),De(t,"onDrag",Et)(A()))}function V(){var q=l[this.a].element,L=l[this.b].element,D=q[Vs](),H=L[Vs]();this.size=D[r]+H[r]+this[Ct]+this[Qt],this.start=D[a],this.end=D[o]}function _(q){if(!getComputedStyle)return null;var L=getComputedStyle(q);if(!L)return null;var D=q[s];return D===0?null:(N===Ta?D-=parseFloat(L.paddingLeft)+parseFloat(L.paddingRight):D-=parseFloat(L.paddingTop)+parseFloat(L.paddingBottom),D)}function B(q){var L=_(d);if(L===null||v.reduce(function(ve,Be){return ve+Be},0)>L)return q;var D=0,H=[],Ee=q.map(function(ve,Be){var Ot=L*ve/100,jn=Pa(x,Be===0,Be===q.length-1,C),Rn=v[Be]+jn;return Ot<Rn?(D+=Rn-Ot,H.push(0),Rn):(H.push(Ot-Rn),Ot)});return D===0?q:Ee.map(function(ve,Be){var Ot=ve;if(D>0&&H[Be]-D>0){var jn=Math.min(D,H[Be]-D);D-=jn,Ot=ve-jn}return Ot/L*100})}function Q(){var q=this,L=l[q.a].element,D=l[q.b].element;q.dragging&&De(t,"onDragEnd",Et)(A()),q.dragging=!1,pt[St]("mouseup",q.stop),pt[St]("touchend",q.stop),pt[St]("touchcancel",q.stop),pt[St]("mousemove",q.move),pt[St]("touchmove",q.move),q.stop=null,q.move=null,L[St]("selectstart",Et),L[St]("dragstart",Et),D[St]("selectstart",Et),D[St]("dragstart",Et),L.style.userSelect="",L.style.webkitUserSelect="",L.style.MozUserSelect="",L.style.pointerEvents="",D.style.userSelect="",D.style.webkitUserSelect="",D.style.MozUserSelect="",D.style.pointerEvents="",q.gutter.style.cursor="",q.parent.style.cursor="",$i.body.style.cursor=""}function w(q){if(!("button"in q&&q.button!==0)){var L=this,D=l[L.a].element,H=l[L.b].element;L.dragging||De(t,"onDragStart",Et)(A()),q.preventDefault(),L.dragging=!0,L.move=O.bind(L),L.stop=Q.bind(L),pt[kt]("mouseup",L.stop),pt[kt]("touchend",L.stop),pt[kt]("touchcancel",L.stop),pt[kt]("mousemove",L.move),pt[kt]("touchmove",L.move),D[kt]("selectstart",Et),D[kt]("dragstart",Et),H[kt]("selectstart",Et),H[kt]("dragstart",Et),D.style.userSelect="none",D.style.webkitUserSelect="none",D.style.MozUserSelect="none",D.style.pointerEvents="none",H.style.userSelect="none",H.style.webkitUserSelect="none",H.style.MozUserSelect="none",H.style.pointerEvents="none",L.gutter.style.cursor=F,L.parent.style.cursor=F,$i.body.style.cursor=F,V.call(L),L.dragOffset=$(q)-L.end}}f=B(f);var ie=[];l=n.map(function(q,L){var D={element:Vp(q),size:f[L],minSize:v[L],maxSize:m[L],snapOffset:S[L],i:L},H;if(L>0&&(H={a:L-1,b:L,dragging:!1,direction:N,parent:d},H[Ct]=Pa(x,L-1===0,!1,C),H[Qt]=Pa(x,!1,L===n.length-1,C),p==="row-reverse"||p==="column-reverse")){var Ee=H.a;H.a=H.b,H.b=Ee}if(L>0){var ve=b(L,N,D.element);ee(ve,x,L),H[oi]=w.bind(H),ve[kt]("mousedown",H[oi]),ve[kt]("touchstart",H[oi]),d.insertBefore(ve,D.element),H.gutter=ve}return X(D.element,D.size,Pa(x,L===0,L===n.length-1,C),L),L>0&&ie.push(H),D});function be(q){var L=q.i===ie.length,D=L?ie[q.i-1]:ie[q.i];V.call(D);var H=L?D.size-q.minSize-D[Qt]:q.minSize+D[Ct];ne.call(D,H)}l.forEach(function(q){var L=q.element[Vs]()[r];L<q.minSize&&(g?be(q):q.minSize=L)});function le(q){var L=B(q);L.forEach(function(D,H){if(H>0){var Ee=ie[H-1],ve=l[Ee.a],Be=l[Ee.b];ve.size=L[H-1],Be.size=D,X(ve.element,ve.size,Ee[Ct],ve.i),X(Be.element,Be.size,Ee[Qt],Be.i)}})}function dt(q,L){ie.forEach(function(D){if(L!==!0?D.parent.removeChild(D.gutter):(D.gutter[St]("mousedown",D[oi]),D.gutter[St]("touchstart",D[oi])),q!==!0){var H=I(r,D.a.size,D[Ct]);Object.keys(H).forEach(function(Ee){l[D.a].element.style[Ee]="",l[D.b].element.style[Ee]=""})}})}return{setSizes:le,getSizes:A,collapse:function(L){be(l[L])},destroy:dt,parent:d,pairs:ie}};function Hs(e,t){var n={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&t.indexOf(r)===-1&&(n[r]=e[r]);return n}var Pr=function(e){function t(){e.apply(this,arguments)}return e&&(t.__proto__=e),t.prototype=Object.create(e&&e.prototype),t.prototype.constructor=t,t.prototype.componentDidMount=function(){var r=this.props;r.children;var i=r.gutter,a=Hs(r,["children","gutter"]),o=a;o.gutter=function(s,l){var u;return i?u=i(s,l):(u=document.createElement("div"),u.className="gutter gutter-"+l),u.__isSplitGutter=!0,u},this.split=Hp(this.parent.children,o)},t.prototype.componentDidUpdate=function(r){var i=this,a=this.props;a.children;var o=a.minSize,s=a.sizes,l=a.collapsed,u=Hs(a,["children","minSize","sizes","collapsed"]),d=u,c=r.minSize,p=r.sizes,f=r.collapsed,y=["maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor"],v=y.map(function(g){return i.props[g]!==r[g]}).reduce(function(g,x){return g||x},!1);if(Array.isArray(o)&&Array.isArray(c)){var k=!1;o.forEach(function(g,x){k=k||g!==c[x]}),v=v||k}else Array.isArray(o)||Array.isArray(c)?v=!0:v=v||o!==c;if(v)d.minSize=o,d.sizes=s||this.split.getSizes(),this.split.destroy(!0,!0),d.gutter=function(g,x,C){return C.previousSibling},this.split=Hp(Array.from(this.parent.children).filter(function(g){return!g.__isSplitGutter}),d);else if(s){var m=!1;s.forEach(function(g,x){m=m||g!==p[x]}),m&&this.split.setSizes(this.props.sizes)}Number.isInteger(l)&&(l!==f||v)&&this.split.collapse(l)},t.prototype.componentWillUnmount=function(){this.split.destroy(),delete this.split},t.prototype.render=function(){var r=this,i=this.props;i.sizes,i.minSize,i.maxSize,i.expandToMin,i.gutterSize,i.gutterAlign,i.snapOffset,i.dragInterval,i.direction,i.cursor,i.gutter,i.elementStyle,i.gutterStyle,i.onDrag,i.onDragStart,i.onDragEnd,i.collapsed;var a=i.children,o=Hs(i,["sizes","minSize","maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor","gutter","elementStyle","gutterStyle","onDrag","onDragStart","onDragEnd","collapsed","children"]),s=o;return Ne.createElement("div",Object.assign({},{ref:function(l){r.parent=l}},s),a)},t}(Ne.Component);Pr.propTypes={sizes:oe.arrayOf(oe.number),minSize:oe.oneOfType([oe.number,oe.arrayOf(oe.number)]),maxSize:oe.oneOfType([oe.number,oe.arrayOf(oe.number)]),expandToMin:oe.bool,gutterSize:oe.number,gutterAlign:oe.string,snapOffset:oe.oneOfType([oe.number,oe.arrayOf(oe.number)]),dragInterval:oe.number,direction:oe.string,cursor:oe.string,gutter:oe.func,elementStyle:oe.func,gutterStyle:oe.func,onDrag:oe.func,onDragStart:oe.func,onDragEnd:oe.func,collapsed:oe.number,children:oe.arrayOf(oe.element)};Pr.defaultProps={sizes:void 0,minSize:void 0,maxSize:void 0,expandToMin:void 0,gutterSize:void 0,gutterAlign:void 0,snapOffset:void 0,dragInterval:void 0,direction:void 0,cursor:void 0,gutter:void 0,elementStyle:void 0,gutterStyle:void 0,onDrag:void 0,onDragStart:void 0,onDragEnd:void 0,collapsed:void 0,children:void 0};function Ng(){const[e,t]=E.useState(!1),[n,r]=E.useState(!1),[i,a]=E.useState(null),[o,s]=E.useState([]),l=E.useRef(null),u=E.useRef(!1),d=E.useCallback(v=>{s(k=>[...k,v])},[]),c=E.useCallback(()=>{s([])},[]),p=E.useCallback(async()=>{if(!(l.current||u.current)){u.current=!0,t(!0),a(null);try{window.loadPyodide||await new Promise((k,m)=>{const g=document.createElement("script");g.src="https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js",g.async=!0,g.onload=()=>k(),g.onerror=()=>m(new Error("Failed to load Pyodide")),document.head.appendChild(g)});const v=await window.loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.26.4/full/"});v.setStdout({batched:k=>{k.split(`
`).forEach(g=>{g.trim()&&d(g)})}}),v.setStderr({batched:k=>{k.split(`
`).forEach(g=>{g.trim()&&d(`[Error] ${g}`)})}}),await v.loadPackage(["numpy"]),l.current=v,r(!0),d("Python environment ready with NumPy loaded.")}catch(v){const k=v instanceof Error?v.message:"Failed to initialize Python";a(k),d(`[Error] ${k}`)}finally{t(!1),u.current=!1}}},[d]);E.useEffect(()=>{p()},[p]);const f=E.useCallback(async v=>{if(!l.current)return{success:!1,error:"Python environment not ready"};try{return c(),await l.current.runPythonAsync(v),{success:!0}}catch(k){const m=k instanceof Error?k.message:"Execution error";return d(`[Error] ${m}`),{success:!1,error:m}}},[d,c]),y=E.useCallback(async(v,k,m)=>{if(!l.current)return k.map(x=>({id:x.id,passed:!1,description:x.description,expected:x.expected,actual:"Python environment not ready",hidden:x.hidden}));const g=[];c();try{await l.current.runPythonAsync(v);for(const x of k)try{const C=x.input.includes(m)||x.input.includes(".")||/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(x.input.trim()),z=x.input.trim().startsWith("(lambda");let S;C&&!x.input.startsWith("[")&&(!x.input.startsWith("(")||z)?S=`
import numpy as np
import json

# Evaluate expression directly
result = ${x.input}

# Convert result to string for comparison
if isinstance(result, bool):
    result_str = str(result)
elif isinstance(result, (np.bool_, )):
    result_str = str(bool(result))
elif isinstance(result, np.ndarray):
    result_str = str(result.tolist())
elif isinstance(result, tuple):
    # Convert tuple elements
    converted = []
    for item in result:
        if isinstance(item, bool):
            converted.append(item)
        elif isinstance(item, (np.bool_, )):
            converted.append(bool(item))
        elif isinstance(item, np.ndarray):
            converted.append(item.tolist())
        elif isinstance(item, (np.floating, float)):
            converted.append(round(float(item), 6))
        elif isinstance(item, (np.integer, int)):
            converted.append(int(item))
        else:
            converted.append(item)
    result_str = str(tuple(converted))
elif isinstance(result, (list, dict)):
    result_str = json.dumps(result)
elif isinstance(result, (np.floating, float)):
    result_str = str(round(float(result), 6))
elif isinstance(result, (np.integer, int)):
    result_str = str(int(result))
else:
    result_str = str(result)
result_str
`:S=`
import numpy as np
import json

# Parse input
test_input = ${x.input}

# Call the function
if isinstance(test_input, list):
    result = ${m}(np.array(test_input))
elif isinstance(test_input, tuple):
    result = ${m}(*[np.array(x) if isinstance(x, list) else x for x in test_input])
else:
    result = ${m}(test_input)

# Convert result to string for comparison
if isinstance(result, bool):
    result_str = str(result)
elif isinstance(result, (np.bool_, )):
    result_str = str(bool(result))
elif isinstance(result, np.ndarray):
    result_str = str(result.tolist())
elif isinstance(result, tuple):
    # Convert tuple elements
    converted = []
    for item in result:
        if isinstance(item, bool):
            converted.append(item)
        elif isinstance(item, (np.bool_, )):
            converted.append(bool(item))
        elif isinstance(item, np.ndarray):
            converted.append(item.tolist())
        elif isinstance(item, (np.floating, float)):
            converted.append(round(float(item), 6))
        elif isinstance(item, (np.integer, int)):
            converted.append(int(item))
        else:
            converted.append(item)
    result_str = str(tuple(converted))
elif isinstance(result, (list, dict)):
    result_str = json.dumps(result)
elif isinstance(result, (np.floating, float)):
    result_str = str(round(float(result), 6))
elif isinstance(result, (np.integer, int)):
    result_str = str(int(result))
else:
    result_str = str(result)
result_str
`;const T=await l.current.runPythonAsync(S),N=String(T),F=x.expected.replace(/\s/g,""),b=N.replace(/\s/g,""),I=F===b;g.push({id:x.id,passed:I,description:x.description,expected:x.expected,actual:N,hidden:x.hidden}),d(I?`Test ${x.id}: PASSED`:`Test ${x.id}: FAILED - Expected ${x.expected}, got ${N}`)}catch(C){const z=C instanceof Error?C.message:"Test execution error";g.push({id:x.id,passed:!1,description:x.description,expected:x.expected,actual:z,hidden:x.hidden}),d(`Test ${x.id}: ERROR - ${z}`)}}catch(x){const C=x instanceof Error?x.message:"Code execution error";return d(`[Error] ${C}`),k.map(z=>({id:z.id,passed:!1,description:z.description,expected:z.expected,actual:C,hidden:z.hidden}))}return g},[d,c]);return{isLoading:e,isReady:n,error:i,output:o,runCode:f,runTests:y,clearOutput:c}}function $p(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}function W2(e){if(Array.isArray(e))return e}function B2(e,t,n){return(t=X2(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function U2(e,t){var n=e==null?null:typeof Symbol<"u"&&e[Symbol.iterator]||e["@@iterator"];if(n!=null){var r,i,a,o,s=[],l=!0,u=!1;try{if(a=(n=n.call(e)).next,t!==0)for(;!(l=(r=a.call(n)).done)&&(s.push(r.value),s.length!==t);l=!0);}catch(d){u=!0,i=d}finally{try{if(!l&&n.return!=null&&(o=n.return(),Object(o)!==o))return}finally{if(u)throw i}}return s}}function V2(){throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function Qp(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function Kp(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?Qp(Object(n),!0).forEach(function(r){B2(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):Qp(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function H2(e,t){if(e==null)return{};var n,r,i=$2(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)===-1&&{}.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}function $2(e,t){if(e==null)return{};var n={};for(var r in e)if({}.hasOwnProperty.call(e,r)){if(t.indexOf(r)!==-1)continue;n[r]=e[r]}return n}function Q2(e,t){return W2(e)||U2(e,t)||Y2(e,t)||V2()}function K2(e,t){if(typeof e!="object"||!e)return e;var n=e[Symbol.toPrimitive];if(n!==void 0){var r=n.call(e,t);if(typeof r!="object")return r;throw new TypeError("@@toPrimitive must return a primitive value.")}return(t==="string"?String:Number)(e)}function X2(e){var t=K2(e,"string");return typeof t=="symbol"?t:t+""}function Y2(e,t){if(e){if(typeof e=="string")return $p(e,t);var n={}.toString.call(e).slice(8,-1);return n==="Object"&&e.constructor&&(n=e.constructor.name),n==="Map"||n==="Set"?Array.from(e):n==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?$p(e,t):void 0}}function G2(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function Xp(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function Yp(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?Xp(Object(n),!0).forEach(function(r){G2(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):Xp(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function Z2(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(r){return t.reduceRight(function(i,a){return a(i)},r)}}function pi(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),a=0;a<r;a++)i[a]=arguments[a];return i.length>=e.length?e.apply(this,i):function(){for(var o=arguments.length,s=new Array(o),l=0;l<o;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function Eo(e){return{}.toString.call(e).includes("Object")}function J2(e){return!Object.keys(e).length}function Qi(e){return typeof e=="function"}function eS(e,t){return Object.prototype.hasOwnProperty.call(e,t)}function tS(e,t){return Eo(t)||Sn("changeType"),Object.keys(t).some(function(n){return!eS(e,n)})&&Sn("changeField"),t}function nS(e){Qi(e)||Sn("selectorType")}function rS(e){Qi(e)||Eo(e)||Sn("handlerType"),Eo(e)&&Object.values(e).some(function(t){return!Qi(t)})&&Sn("handlersType")}function iS(e){e||Sn("initialIsRequired"),Eo(e)||Sn("initialType"),J2(e)&&Sn("initialContent")}function aS(e,t){throw new Error(e[t]||e.default)}var oS={initialIsRequired:"initial state is required",initialType:"initial state should be an object",initialContent:"initial state shouldn't be an empty object",handlerType:"handler should be an object or a function",handlersType:"all handlers should be a functions",selectorType:"selector should be a function",changeType:"provided value of changes should be an object",changeField:'it seams you want to change a field in the state which is not specified in the "initial" state',default:"an unknown error accured in `state-local` package"},Sn=pi(aS)(oS),Na={changes:tS,selector:nS,handler:rS,initial:iS};function sS(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};Na.initial(e),Na.handler(t);var n={current:e},r=pi(cS)(n,t),i=pi(uS)(n),a=pi(Na.changes)(e),o=pi(lS)(n);function s(){var u=arguments.length>0&&arguments[0]!==void 0?arguments[0]:function(d){return d};return Na.selector(u),u(n.current)}function l(u){Z2(r,i,a,o)(u)}return[s,l]}function lS(e,t){return Qi(t)?t(e.current):t}function uS(e,t){return e.current=Yp(Yp({},e.current),t),t}function cS(e,t,n){return Qi(t)?t(e.current):Object.keys(n).forEach(function(r){var i;return(i=t[r])===null||i===void 0?void 0:i.call(t,e.current[r])}),n}var dS={create:sS},pS={paths:{vs:"https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs"}};function fS(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),a=0;a<r;a++)i[a]=arguments[a];return i.length>=e.length?e.apply(this,i):function(){for(var o=arguments.length,s=new Array(o),l=0;l<o;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function mS(e){return{}.toString.call(e).includes("Object")}function hS(e){return e||Gp("configIsRequired"),mS(e)||Gp("configType"),e.urls?(gS(),{paths:{vs:e.urls.monacoBase}}):e}function gS(){console.warn(Ag.deprecation)}function yS(e,t){throw new Error(e[t]||e.default)}var Ag={configIsRequired:"the configuration object is required",configType:"the configuration object should be an object",default:"an unknown error accured in `@monaco-editor/loader` package",deprecation:`Deprecation warning!
    You are using deprecated way of configuration.

    Instead of using
      monaco.config({ urls: { monacoBase: '...' } })
    use
      monaco.config({ paths: { vs: '...' } })

    For more please check the link https://github.com/suren-atoyan/monaco-loader#config
  `},Gp=fS(yS)(Ag),xS={config:hS},vS=function(){for(var t=arguments.length,n=new Array(t),r=0;r<t;r++)n[r]=arguments[r];return function(i){return n.reduceRight(function(a,o){return o(a)},i)}};function Lg(e,t){return Object.keys(t).forEach(function(n){t[n]instanceof Object&&e[n]&&Object.assign(t[n],Lg(e[n],t[n]))}),Kp(Kp({},e),t)}var _S={type:"cancelation",msg:"operation is manually canceled"};function $s(e){var t=!1,n=new Promise(function(r,i){e.then(function(a){return t?i(_S):r(a)}),e.catch(i)});return n.cancel=function(){return t=!0},n}var wS=["monaco"],bS=dS.create({config:pS,isInitialized:!1,resolve:null,reject:null,monaco:null}),jg=Q2(bS,2),oa=jg[0],es=jg[1];function kS(e){var t=xS.config(e),n=t.monaco,r=H2(t,wS);es(function(i){return{config:Lg(i.config,r),monaco:n}})}function SS(){var e=oa(function(t){var n=t.monaco,r=t.isInitialized,i=t.resolve;return{monaco:n,isInitialized:r,resolve:i}});if(!e.isInitialized){if(es({isInitialized:!0}),e.monaco)return e.resolve(e.monaco),$s(Qs);if(window.monaco&&window.monaco.editor)return Rg(window.monaco),e.resolve(window.monaco),$s(Qs);vS(CS,zS)(TS)}return $s(Qs)}function CS(e){return document.body.appendChild(e)}function ES(e){var t=document.createElement("script");return e&&(t.src=e),t}function zS(e){var t=oa(function(r){var i=r.config,a=r.reject;return{config:i,reject:a}}),n=ES("".concat(t.config.paths.vs,"/loader.js"));return n.onload=function(){return e()},n.onerror=t.reject,n}function TS(){var e=oa(function(n){var r=n.config,i=n.resolve,a=n.reject;return{config:r,resolve:i,reject:a}}),t=window.require;t.config(e.config),t(["vs/editor/editor.main"],function(n){var r=n.m||n;Rg(r),e.resolve(r)},function(n){e.reject(n)})}function Rg(e){oa().monaco||es({monaco:e})}function PS(){return oa(function(e){var t=e.monaco;return t})}var Qs=new Promise(function(e,t){return es({resolve:e,reject:t})}),Ig={config:kS,init:SS,__getMonacoInstance:PS},NS={wrapper:{display:"flex",position:"relative",textAlign:"initial"},fullWidth:{width:"100%"},hide:{display:"none"}},Ks=NS,AS={container:{display:"flex",height:"100%",width:"100%",justifyContent:"center",alignItems:"center"}},LS=AS;function jS({children:e}){return Ne.createElement("div",{style:LS.container},e)}var RS=jS,IS=RS;function OS({width:e,height:t,isEditorReady:n,loading:r,_ref:i,className:a,wrapperProps:o}){return Ne.createElement("section",{style:{...Ks.wrapper,width:e,height:t},...o},!n&&Ne.createElement(IS,null,r),Ne.createElement("div",{ref:i,style:{...Ks.fullWidth,...!n&&Ks.hide},className:a}))}var DS=OS,Og=E.memo(DS);function MS(e){E.useEffect(e,[])}var Dg=MS;function FS(e,t,n=!0){let r=E.useRef(!0);E.useEffect(r.current||!n?()=>{r.current=!1}:e,t)}var it=FS;function Ci(){}function vr(e,t,n,r){return qS(e,r)||WS(e,t,n,r)}function qS(e,t){return e.editor.getModel(Mg(e,t))}function WS(e,t,n,r){return e.editor.createModel(t,n,r?Mg(e,r):void 0)}function Mg(e,t){return e.Uri.parse(t)}function BS({original:e,modified:t,language:n,originalLanguage:r,modifiedLanguage:i,originalModelPath:a,modifiedModelPath:o,keepCurrentOriginalModel:s=!1,keepCurrentModifiedModel:l=!1,theme:u="light",loading:d="Loading...",options:c={},height:p="100%",width:f="100%",className:y,wrapperProps:v={},beforeMount:k=Ci,onMount:m=Ci}){let[g,x]=E.useState(!1),[C,z]=E.useState(!0),S=E.useRef(null),T=E.useRef(null),N=E.useRef(null),F=E.useRef(m),b=E.useRef(k),I=E.useRef(!1);Dg(()=>{let A=Ig.init();return A.then($=>(T.current=$)&&z(!1)).catch($=>($==null?void 0:$.type)!=="cancelation"&&console.error("Monaco initialization: error:",$)),()=>S.current?ee():A.cancel()}),it(()=>{if(S.current&&T.current){let A=S.current.getOriginalEditor(),$=vr(T.current,e||"",r||n||"text",a||"");$!==A.getModel()&&A.setModel($)}},[a],g),it(()=>{if(S.current&&T.current){let A=S.current.getModifiedEditor(),$=vr(T.current,t||"",i||n||"text",o||"");$!==A.getModel()&&A.setModel($)}},[o],g),it(()=>{let A=S.current.getModifiedEditor();A.getOption(T.current.editor.EditorOption.readOnly)?A.setValue(t||""):t!==A.getValue()&&(A.executeEdits("",[{range:A.getModel().getFullModelRange(),text:t||"",forceMoveMarkers:!0}]),A.pushUndoStop())},[t],g),it(()=>{var A,$;($=(A=S.current)==null?void 0:A.getModel())==null||$.original.setValue(e||"")},[e],g),it(()=>{let{original:A,modified:$}=S.current.getModel();T.current.editor.setModelLanguage(A,r||n||"text"),T.current.editor.setModelLanguage($,i||n||"text")},[n,r,i],g),it(()=>{var A;(A=T.current)==null||A.editor.setTheme(u)},[u],g),it(()=>{var A;(A=S.current)==null||A.updateOptions(c)},[c],g);let W=E.useCallback(()=>{var ne;if(!T.current)return;b.current(T.current);let A=vr(T.current,e||"",r||n||"text",a||""),$=vr(T.current,t||"",i||n||"text",o||"");(ne=S.current)==null||ne.setModel({original:A,modified:$})},[n,t,i,e,r,a,o]),X=E.useCallback(()=>{var A;!I.current&&N.current&&(S.current=T.current.editor.createDiffEditor(N.current,{automaticLayout:!0,...c}),W(),(A=T.current)==null||A.editor.setTheme(u),x(!0),I.current=!0)},[c,u,W]);E.useEffect(()=>{g&&F.current(S.current,T.current)},[g]),E.useEffect(()=>{!C&&!g&&X()},[C,g,X]);function ee(){var $,ne,O,V;let A=($=S.current)==null?void 0:$.getModel();s||((ne=A==null?void 0:A.original)==null||ne.dispose()),l||((O=A==null?void 0:A.modified)==null||O.dispose()),(V=S.current)==null||V.dispose()}return Ne.createElement(Og,{width:f,height:p,isEditorReady:g,loading:d,_ref:N,className:y,wrapperProps:v})}var US=BS;E.memo(US);function VS(e){let t=E.useRef();return E.useEffect(()=>{t.current=e},[e]),t.current}var HS=VS,Aa=new Map;function $S({defaultValue:e,defaultLanguage:t,defaultPath:n,value:r,language:i,path:a,theme:o="light",line:s,loading:l="Loading...",options:u={},overrideServices:d={},saveViewState:c=!0,keepCurrentModel:p=!1,width:f="100%",height:y="100%",className:v,wrapperProps:k={},beforeMount:m=Ci,onMount:g=Ci,onChange:x,onValidate:C=Ci}){let[z,S]=E.useState(!1),[T,N]=E.useState(!0),F=E.useRef(null),b=E.useRef(null),I=E.useRef(null),W=E.useRef(g),X=E.useRef(m),ee=E.useRef(),A=E.useRef(r),$=HS(a),ne=E.useRef(!1),O=E.useRef(!1);Dg(()=>{let B=Ig.init();return B.then(Q=>(F.current=Q)&&N(!1)).catch(Q=>(Q==null?void 0:Q.type)!=="cancelation"&&console.error("Monaco initialization: error:",Q)),()=>b.current?_():B.cancel()}),it(()=>{var Q,w,ie,be;let B=vr(F.current,e||r||"",t||i||"",a||n||"");B!==((Q=b.current)==null?void 0:Q.getModel())&&(c&&Aa.set($,(w=b.current)==null?void 0:w.saveViewState()),(ie=b.current)==null||ie.setModel(B),c&&((be=b.current)==null||be.restoreViewState(Aa.get(a))))},[a],z),it(()=>{var B;(B=b.current)==null||B.updateOptions(u)},[u],z),it(()=>{!b.current||r===void 0||(b.current.getOption(F.current.editor.EditorOption.readOnly)?b.current.setValue(r):r!==b.current.getValue()&&(O.current=!0,b.current.executeEdits("",[{range:b.current.getModel().getFullModelRange(),text:r,forceMoveMarkers:!0}]),b.current.pushUndoStop(),O.current=!1))},[r],z),it(()=>{var Q,w;let B=(Q=b.current)==null?void 0:Q.getModel();B&&i&&((w=F.current)==null||w.editor.setModelLanguage(B,i))},[i],z),it(()=>{var B;s!==void 0&&((B=b.current)==null||B.revealLine(s))},[s],z),it(()=>{var B;(B=F.current)==null||B.editor.setTheme(o)},[o],z);let V=E.useCallback(()=>{var B;if(!(!I.current||!F.current)&&!ne.current){X.current(F.current);let Q=a||n,w=vr(F.current,r||e||"",t||i||"",Q||"");b.current=(B=F.current)==null?void 0:B.editor.create(I.current,{model:w,automaticLayout:!0,...u},d),c&&b.current.restoreViewState(Aa.get(Q)),F.current.editor.setTheme(o),s!==void 0&&b.current.revealLine(s),S(!0),ne.current=!0}},[e,t,n,r,i,a,u,d,c,o,s]);E.useEffect(()=>{z&&W.current(b.current,F.current)},[z]),E.useEffect(()=>{!T&&!z&&V()},[T,z,V]),A.current=r,E.useEffect(()=>{var B,Q;z&&x&&((B=ee.current)==null||B.dispose(),ee.current=(Q=b.current)==null?void 0:Q.onDidChangeModelContent(w=>{O.current||x(b.current.getValue(),w)}))},[z,x]),E.useEffect(()=>{if(z){let B=F.current.editor.onDidChangeMarkers(Q=>{var ie;let w=(ie=b.current.getModel())==null?void 0:ie.uri;if(w&&Q.find(be=>be.path===w.path)){let be=F.current.editor.getModelMarkers({resource:w});C==null||C(be)}});return()=>{B==null||B.dispose()}}return()=>{}},[z,C]);function _(){var B,Q;(B=ee.current)==null||B.dispose(),p?c&&Aa.set(a,b.current.saveViewState()):(Q=b.current.getModel())==null||Q.dispose(),b.current.dispose()}return Ne.createElement(Og,{width:f,height:y,isEditorReady:z,loading:l,_ref:I,className:v,wrapperProps:k})}var QS=$S,KS=E.memo(QS),XS=KS;function pu({value:e,onChange:t,height:n="400px",onRun:r}){const i=E.useRef(null),a=E.useRef(r);E.useEffect(()=>{a.current=r},[r]);const o=E.useCallback(l=>{t(l||"")},[t]),s=E.useCallback((l,u)=>{i.current=l,l.addAction({id:"run-tests",label:"Run Tests",keybindings:[u.KeyMod.Shift|u.KeyCode.Enter],run:()=>{a.current&&a.current()}})},[]);return h.jsx("div",{className:"h-full border border-gray-200 rounded-lg overflow-hidden shadow-sm",children:h.jsx(XS,{height:n,defaultLanguage:"python",theme:"light",value:e,onChange:o,onMount:s,options:{minimap:{enabled:!1},fontSize:14,lineNumbers:"on",scrollBeyondLastLine:!1,automaticLayout:!0,tabSize:4,wordWrap:"on",padding:{top:16,bottom:16},scrollbar:{verticalScrollbarSize:8,horizontalScrollbarSize:8}}})})}function YS(e){const t=e.filter(n=>!n.hidden);return t.length===0?"":t.map((n,r)=>`# Test ${r+1}: ${n.description}
input: ${n.input}
expected: ${n.expected}`).join(`

`)}function GS(e,t){const n=t.filter(a=>a.hidden);return e.trim()?[...e.split(/\n\n+/).filter(a=>a.trim()).map((a,o)=>{const s=a.split(`
`);let l=`Test ${o+1}`,u="",d="";for(const c of s)if(c.startsWith("# Test")){const p=c.match(/# Test \d+: (.+)/);p&&(l=p[1])}else c.startsWith("input:")?u=c.replace("input:","").trim():c.startsWith("expected:")&&(d=c.replace("expected:","").trim());return{id:String(o+1),description:l,input:u,expected:d,hidden:!1}}).filter(a=>a.input&&a.expected),...n]:n}function fu({output:e,isLoading:t=!1,testCases:n,onTestCasesChange:r,problemId:i,resetKey:a=0}){const o=E.useRef(null),[s,l]=E.useState("tests"),[u,d]=E.useState(""),c=E.useRef(void 0),p=E.useRef(0),f=(n==null?void 0:n.filter(m=>!m.hidden).length)??0,y=(n==null?void 0:n.filter(m=>m.hidden).length)??0;E.useEffect(()=>{if(n&&n.length>0){const m=i!==c.current,g=a!==p.current,x=n.filter(z=>!z.hidden),C=u===""&&x.length>0;(m||g||C)&&(d(YS(n)),c.current=i,p.current=a,m&&l("tests"))}},[n,i,a,u]),E.useEffect(()=>{e.length>0&&l("output")},[e]),E.useEffect(()=>{var m;s==="output"&&((m=o.current)==null||m.scrollIntoView({behavior:"smooth"}))},[e,s]);const v=m=>{if(d(m),r&&n){const g=GS(m,n);r(g)}},k=n&&f>0;return h.jsxs("div",{className:"bg-white border border-gray-200 rounded-lg h-full flex flex-col shadow-sm",children:[h.jsxs("div",{className:"flex items-center justify-between px-2 py-1 border-b border-gray-200 bg-gray-50",children:[h.jsxs("div",{className:"flex",children:[k&&h.jsxs("button",{onClick:()=>l("tests"),className:`px-3 py-1.5 text-sm font-medium rounded-t transition-colors ${s==="tests"?"text-primary-600 bg-white border-t border-l border-r border-gray-200 -mb-px":"text-gray-500 hover:text-gray-700"}`,children:["Tests (",f,")"]}),h.jsx("button",{onClick:()=>l("output"),className:`px-3 py-1.5 text-sm font-medium rounded-t transition-colors ${s==="output"?"text-primary-600 bg-white border-t border-l border-r border-gray-200 -mb-px":"text-gray-500 hover:text-gray-700"}`,children:"Output"})]}),t&&h.jsxs("div",{className:"flex items-center gap-2",children:[h.jsx("div",{className:"w-2 h-2 bg-primary-500 rounded-full animate-pulse"}),h.jsx("span",{className:"text-xs text-gray-500",children:"Running..."})]})]}),s==="tests"&&k?h.jsxs("div",{className:"flex-1 flex flex-col overflow-hidden",children:[h.jsx("textarea",{value:u,onChange:m=>v(m.target.value),className:"flex-1 p-4 font-mono text-sm bg-gray-900 text-gray-300 resize-none focus:outline-none",placeholder:`# Test 1: Description
input: your_function(args)
expected: expected_result`,spellCheck:!1}),y>0&&h.jsxs("div",{className:"px-4 py-2 bg-gray-800 text-gray-400 text-xs border-t border-gray-700",children:["+ ",y," hidden test",y>1?"s":""," (not editable)"]})]}):h.jsxs("div",{className:"flex-1 overflow-auto p-4 font-mono text-sm bg-gray-900 rounded-b-lg",children:[e.length===0?h.jsx("div",{className:"text-gray-500 italic",children:"Output will appear here when you run your code..."}):e.map((m,g)=>h.jsx("div",{className:`py-0.5 ${m.startsWith("[Error]")?"text-red-400":m.includes("PASSED")?"text-green-400":m.includes("FAILED")?"text-red-400":"text-gray-300"}`,children:m.startsWith("Test")?h.jsxs("span",{children:[m.includes("PASSED")?"✓ ":m.includes("FAILED")?"✗ ":"",m]}):h.jsxs("span",{children:["> ",m]})},g)),h.jsx("div",{ref:o})]})]})}function Zp(e,t){const n=String(e);if(typeof t!="string")throw new TypeError("Expected character");let r=0,i=n.indexOf(t);for(;i!==-1;)r++,i=n.indexOf(t,i+t.length);return r}function ZS(e){if(typeof e!="string")throw new TypeError("Expected a string");return e.replace(/[|\\{}()[\]^$+*?.]/g,"\\$&").replace(/-/g,"\\x2d")}function JS(e,t,n){const i=Zo((n||{}).ignore||[]),a=eC(t);let o=-1;for(;++o<a.length;)wg(e,"text",s);function s(u,d){let c=-1,p;for(;++c<d.length;){const f=d[c],y=p?p.children:void 0;if(i(f,y?y.indexOf(f):void 0,p))return;p=f}if(p)return l(u,d)}function l(u,d){const c=d[d.length-1],p=a[o][0],f=a[o][1];let y=0;const k=c.children.indexOf(u);let m=!1,g=[];p.lastIndex=0;let x=p.exec(u.value);for(;x;){const C=x.index,z={index:x.index,input:x.input,stack:[...d,u]};let S=f(...x,z);if(typeof S=="string"&&(S=S.length>0?{type:"text",value:S}:void 0),S===!1?p.lastIndex=C+1:(y!==C&&g.push({type:"text",value:u.value.slice(y,C)}),Array.isArray(S)?g.push(...S):S&&g.push(S),y=C+x[0].length,m=!0),!p.global)break;x=p.exec(u.value)}return m?(y<u.value.length&&g.push({type:"text",value:u.value.slice(y)}),c.children.splice(k,1,...g)):g=[u],k+g.length}}function eC(e){const t=[];if(!Array.isArray(e))throw new TypeError("Expected find and replace tuple or list of tuples");const n=!e[0]||Array.isArray(e[0])?e:[e];let r=-1;for(;++r<n.length;){const i=n[r];t.push([tC(i[0]),nC(i[1])])}return t}function tC(e){return typeof e=="string"?new RegExp(ZS(e),"g"):e}function nC(e){return typeof e=="function"?e:function(){return e}}const Xs="phrasing",Ys=["autolink","link","image","label"];function rC(){return{transforms:[cC],enter:{literalAutolink:aC,literalAutolinkEmail:Gs,literalAutolinkHttp:Gs,literalAutolinkWww:Gs},exit:{literalAutolink:uC,literalAutolinkEmail:lC,literalAutolinkHttp:oC,literalAutolinkWww:sC}}}function iC(){return{unsafe:[{character:"@",before:"[+\\-.\\w]",after:"[\\-.\\w]",inConstruct:Xs,notInConstruct:Ys},{character:".",before:"[Ww]",after:"[\\-.\\w]",inConstruct:Xs,notInConstruct:Ys},{character:":",before:"[ps]",after:"\\/",inConstruct:Xs,notInConstruct:Ys}]}}function aC(e){this.enter({type:"link",title:null,url:"",children:[]},e)}function Gs(e){this.config.enter.autolinkProtocol.call(this,e)}function oC(e){this.config.exit.autolinkProtocol.call(this,e)}function sC(e){this.config.exit.data.call(this,e);const t=this.stack[this.stack.length-1];t.type,t.url="http://"+this.sliceSerialize(e)}function lC(e){this.config.exit.autolinkEmail.call(this,e)}function uC(e){this.exit(e)}function cC(e){JS(e,[[/(https?:\/\/|www(?=\.))([-.\w]+)([^ \t\r\n]*)/gi,dC],[new RegExp("(?<=^|\\s|\\p{P}|\\p{S})([-.\\w+]+)@([-\\w]+(?:\\.[-\\w]+)+)","gu"),pC]],{ignore:["link","linkReference"]})}function dC(e,t,n,r,i){let a="";if(!Fg(i)||(/^w/i.test(t)&&(n=t+n,t="",a="http://"),!fC(n)))return!1;const o=mC(n+r);if(!o[0])return!1;const s={type:"link",title:null,url:a+t+o[0],children:[{type:"text",value:t+o[0]}]};return o[1]?[s,{type:"text",value:o[1]}]:s}function pC(e,t,n,r){return!Fg(r,!0)||/[-\d_]$/.test(n)?!1:{type:"link",title:null,url:"mailto:"+t+"@"+n,children:[{type:"text",value:t+"@"+n}]}}function fC(e){const t=e.split(".");return!(t.length<2||t[t.length-1]&&(/_/.test(t[t.length-1])||!/[a-zA-Z\d]/.test(t[t.length-1]))||t[t.length-2]&&(/_/.test(t[t.length-2])||!/[a-zA-Z\d]/.test(t[t.length-2])))}function mC(e){const t=/[!"&'),.:;<>?\]}]+$/.exec(e);if(!t)return[e,void 0];e=e.slice(0,t.index);let n=t[0],r=n.indexOf(")");const i=Zp(e,"(");let a=Zp(e,")");for(;r!==-1&&i>a;)e+=n.slice(0,r+1),n=n.slice(r+1),r=n.indexOf(")"),a++;return[e,n]}function Fg(e,t){const n=e.input.charCodeAt(e.index-1);return(e.index===0||Gn(n)||Xo(n))&&(!t||n!==47)}qg.peek=kC;function hC(){this.buffer()}function gC(e){this.enter({type:"footnoteReference",identifier:"",label:""},e)}function yC(){this.buffer()}function xC(e){this.enter({type:"footnoteDefinition",identifier:"",label:"",children:[]},e)}function vC(e){const t=this.resume(),n=this.stack[this.stack.length-1];n.type,n.identifier=Rt(this.sliceSerialize(e)).toLowerCase(),n.label=t}function _C(e){this.exit(e)}function wC(e){const t=this.resume(),n=this.stack[this.stack.length-1];n.type,n.identifier=Rt(this.sliceSerialize(e)).toLowerCase(),n.label=t}function bC(e){this.exit(e)}function kC(){return"["}function qg(e,t,n,r){const i=n.createTracker(r);let a=i.move("[^");const o=n.enter("footnoteReference"),s=n.enter("reference");return a+=i.move(n.safe(n.associationId(e),{after:"]",before:a})),s(),o(),a+=i.move("]"),a}function SC(){return{enter:{gfmFootnoteCallString:hC,gfmFootnoteCall:gC,gfmFootnoteDefinitionLabelString:yC,gfmFootnoteDefinition:xC},exit:{gfmFootnoteCallString:vC,gfmFootnoteCall:_C,gfmFootnoteDefinitionLabelString:wC,gfmFootnoteDefinition:bC}}}function CC(e){let t=!1;return e&&e.firstLineBlank&&(t=!0),{handlers:{footnoteDefinition:n,footnoteReference:qg},unsafe:[{character:"[",inConstruct:["label","phrasing","reference"]}]};function n(r,i,a,o){const s=a.createTracker(o);let l=s.move("[^");const u=a.enter("footnoteDefinition"),d=a.enter("label");return l+=s.move(a.safe(a.associationId(r),{before:l,after:"]"})),d(),l+=s.move("]:"),r.children&&r.children.length>0&&(s.shift(4),l+=s.move((t?`
`:" ")+a.indentLines(a.containerFlow(r,s.current()),t?Wg:EC))),u(),l}}function EC(e,t,n){return t===0?e:Wg(e,t,n)}function Wg(e,t,n){return(n?"":"    ")+e}const zC=["autolink","destinationLiteral","destinationRaw","reference","titleQuote","titleApostrophe"];Bg.peek=LC;function TC(){return{canContainEols:["delete"],enter:{strikethrough:NC},exit:{strikethrough:AC}}}function PC(){return{unsafe:[{character:"~",inConstruct:"phrasing",notInConstruct:zC}],handlers:{delete:Bg}}}function NC(e){this.enter({type:"delete",children:[]},e)}function AC(e){this.exit(e)}function Bg(e,t,n,r){const i=n.createTracker(r),a=n.enter("strikethrough");let o=i.move("~~");return o+=n.containerPhrasing(e,{...i.current(),before:o,after:"~"}),o+=i.move("~~"),a(),o}function LC(){return"~"}function jC(e){return e.length}function RC(e,t){const n=t||{},r=(n.align||[]).concat(),i=n.stringLength||jC,a=[],o=[],s=[],l=[];let u=0,d=-1;for(;++d<e.length;){const v=[],k=[];let m=-1;for(e[d].length>u&&(u=e[d].length);++m<e[d].length;){const g=IC(e[d][m]);if(n.alignDelimiters!==!1){const x=i(g);k[m]=x,(l[m]===void 0||x>l[m])&&(l[m]=x)}v.push(g)}o[d]=v,s[d]=k}let c=-1;if(typeof r=="object"&&"length"in r)for(;++c<u;)a[c]=Jp(r[c]);else{const v=Jp(r);for(;++c<u;)a[c]=v}c=-1;const p=[],f=[];for(;++c<u;){const v=a[c];let k="",m="";v===99?(k=":",m=":"):v===108?k=":":v===114&&(m=":");let g=n.alignDelimiters===!1?1:Math.max(1,l[c]-k.length-m.length);const x=k+"-".repeat(g)+m;n.alignDelimiters!==!1&&(g=k.length+g+m.length,g>l[c]&&(l[c]=g),f[c]=g),p[c]=x}o.splice(1,0,p),s.splice(1,0,f),d=-1;const y=[];for(;++d<o.length;){const v=o[d],k=s[d];c=-1;const m=[];for(;++c<u;){const g=v[c]||"";let x="",C="";if(n.alignDelimiters!==!1){const z=l[c]-(k[c]||0),S=a[c];S===114?x=" ".repeat(z):S===99?z%2?(x=" ".repeat(z/2+.5),C=" ".repeat(z/2-.5)):(x=" ".repeat(z/2),C=x):C=" ".repeat(z)}n.delimiterStart!==!1&&!c&&m.push("|"),n.padding!==!1&&!(n.alignDelimiters===!1&&g==="")&&(n.delimiterStart!==!1||c)&&m.push(" "),n.alignDelimiters!==!1&&m.push(x),m.push(g),n.alignDelimiters!==!1&&m.push(C),n.padding!==!1&&m.push(" "),(n.delimiterEnd!==!1||c!==u-1)&&m.push("|")}y.push(n.delimiterEnd===!1?m.join("").replace(/ +$/,""):m.join(""))}return y.join(`
`)}function IC(e){return e==null?"":String(e)}function Jp(e){const t=typeof e=="string"?e.codePointAt(0):0;return t===67||t===99?99:t===76||t===108?108:t===82||t===114?114:0}function OC(e,t,n,r){const i=n.enter("blockquote"),a=n.createTracker(r);a.move("> "),a.shift(2);const o=n.indentLines(n.containerFlow(e,a.current()),DC);return i(),o}function DC(e,t,n){return">"+(n?"":" ")+e}function MC(e,t){return ef(e,t.inConstruct,!0)&&!ef(e,t.notInConstruct,!1)}function ef(e,t,n){if(typeof t=="string"&&(t=[t]),!t||t.length===0)return n;let r=-1;for(;++r<t.length;)if(e.includes(t[r]))return!0;return!1}function tf(e,t,n,r){let i=-1;for(;++i<n.unsafe.length;)if(n.unsafe[i].character===`
`&&MC(n.stack,n.unsafe[i]))return/[ \t]/.test(r.before)?"":" ";return`\\
`}function FC(e,t){const n=String(e);let r=n.indexOf(t),i=r,a=0,o=0;if(typeof t!="string")throw new TypeError("Expected substring");for(;r!==-1;)r===i?++a>o&&(o=a):a=1,i=r+t.length,r=n.indexOf(t,i);return o}function qC(e,t){return!!(t.options.fences===!1&&e.value&&!e.lang&&/[^ \r\n]/.test(e.value)&&!/^[\t ]*(?:[\r\n]|$)|(?:^|[\r\n])[\t ]*$/.test(e.value))}function WC(e){const t=e.options.fence||"`";if(t!=="`"&&t!=="~")throw new Error("Cannot serialize code with `"+t+"` for `options.fence`, expected `` ` `` or `~`");return t}function BC(e,t,n,r){const i=WC(n),a=e.value||"",o=i==="`"?"GraveAccent":"Tilde";if(qC(e,n)){const c=n.enter("codeIndented"),p=n.indentLines(a,UC);return c(),p}const s=n.createTracker(r),l=i.repeat(Math.max(FC(a,i)+1,3)),u=n.enter("codeFenced");let d=s.move(l);if(e.lang){const c=n.enter(`codeFencedLang${o}`);d+=s.move(n.safe(e.lang,{before:d,after:" ",encode:["`"],...s.current()})),c()}if(e.lang&&e.meta){const c=n.enter(`codeFencedMeta${o}`);d+=s.move(" "),d+=s.move(n.safe(e.meta,{before:d,after:`
`,encode:["`"],...s.current()})),c()}return d+=s.move(`
`),a&&(d+=s.move(a+`
`)),d+=s.move(l),u(),d}function UC(e,t,n){return(n?"":"    ")+e}function Tc(e){const t=e.options.quote||'"';if(t!=='"'&&t!=="'")throw new Error("Cannot serialize title with `"+t+"` for `options.quote`, expected `\"`, or `'`");return t}function VC(e,t,n,r){const i=Tc(n),a=i==='"'?"Quote":"Apostrophe",o=n.enter("definition");let s=n.enter("label");const l=n.createTracker(r);let u=l.move("[");return u+=l.move(n.safe(n.associationId(e),{before:u,after:"]",...l.current()})),u+=l.move("]: "),s(),!e.url||/[\0- \u007F]/.test(e.url)?(s=n.enter("destinationLiteral"),u+=l.move("<"),u+=l.move(n.safe(e.url,{before:u,after:">",...l.current()})),u+=l.move(">")):(s=n.enter("destinationRaw"),u+=l.move(n.safe(e.url,{before:u,after:e.title?" ":`
`,...l.current()}))),s(),e.title&&(s=n.enter(`title${a}`),u+=l.move(" "+i),u+=l.move(n.safe(e.title,{before:u,after:i,...l.current()})),u+=l.move(i),s()),o(),u}function HC(e){const t=e.options.emphasis||"*";if(t!=="*"&&t!=="_")throw new Error("Cannot serialize emphasis with `"+t+"` for `options.emphasis`, expected `*`, or `_`");return t}function Ki(e){return"&#x"+e.toString(16).toUpperCase()+";"}function zo(e,t,n){const r=Fr(e),i=Fr(t);return r===void 0?i===void 0?n==="_"?{inside:!0,outside:!0}:{inside:!1,outside:!1}:i===1?{inside:!0,outside:!0}:{inside:!1,outside:!0}:r===1?i===void 0?{inside:!1,outside:!1}:i===1?{inside:!0,outside:!0}:{inside:!1,outside:!1}:i===void 0?{inside:!1,outside:!1}:i===1?{inside:!0,outside:!1}:{inside:!1,outside:!1}}Ug.peek=$C;function Ug(e,t,n,r){const i=HC(n),a=n.enter("emphasis"),o=n.createTracker(r),s=o.move(i);let l=o.move(n.containerPhrasing(e,{after:i,before:s,...o.current()}));const u=l.charCodeAt(0),d=zo(r.before.charCodeAt(r.before.length-1),u,i);d.inside&&(l=Ki(u)+l.slice(1));const c=l.charCodeAt(l.length-1),p=zo(r.after.charCodeAt(0),c,i);p.inside&&(l=l.slice(0,-1)+Ki(c));const f=o.move(i);return a(),n.attentionEncodeSurroundingInfo={after:p.outside,before:d.outside},s+l+f}function $C(e,t,n){return n.options.emphasis||"*"}function QC(e,t){let n=!1;return Cc(e,function(r){if("value"in r&&/\r?\n|\r/.test(r.value)||r.type==="break")return n=!0,lu}),!!((!e.depth||e.depth<3)&&xc(e)&&(t.options.setext||n))}function KC(e,t,n,r){const i=Math.max(Math.min(6,e.depth||1),1),a=n.createTracker(r);if(QC(e,n)){const d=n.enter("headingSetext"),c=n.enter("phrasing"),p=n.containerPhrasing(e,{...a.current(),before:`
`,after:`
`});return c(),d(),p+`
`+(i===1?"=":"-").repeat(p.length-(Math.max(p.lastIndexOf("\r"),p.lastIndexOf(`
`))+1))}const o="#".repeat(i),s=n.enter("headingAtx"),l=n.enter("phrasing");a.move(o+" ");let u=n.containerPhrasing(e,{before:"# ",after:`
`,...a.current()});return/^[\t ]/.test(u)&&(u=Ki(u.charCodeAt(0))+u.slice(1)),u=u?o+" "+u:o,n.options.closeAtx&&(u+=" "+o),l(),s(),u}Vg.peek=XC;function Vg(e){return e.value||""}function XC(){return"<"}Hg.peek=YC;function Hg(e,t,n,r){const i=Tc(n),a=i==='"'?"Quote":"Apostrophe",o=n.enter("image");let s=n.enter("label");const l=n.createTracker(r);let u=l.move("![");return u+=l.move(n.safe(e.alt,{before:u,after:"]",...l.current()})),u+=l.move("]("),s(),!e.url&&e.title||/[\0- \u007F]/.test(e.url)?(s=n.enter("destinationLiteral"),u+=l.move("<"),u+=l.move(n.safe(e.url,{before:u,after:">",...l.current()})),u+=l.move(">")):(s=n.enter("destinationRaw"),u+=l.move(n.safe(e.url,{before:u,after:e.title?" ":")",...l.current()}))),s(),e.title&&(s=n.enter(`title${a}`),u+=l.move(" "+i),u+=l.move(n.safe(e.title,{before:u,after:i,...l.current()})),u+=l.move(i),s()),u+=l.move(")"),o(),u}function YC(){return"!"}$g.peek=GC;function $g(e,t,n,r){const i=e.referenceType,a=n.enter("imageReference");let o=n.enter("label");const s=n.createTracker(r);let l=s.move("![");const u=n.safe(e.alt,{before:l,after:"]",...s.current()});l+=s.move(u+"]["),o();const d=n.stack;n.stack=[],o=n.enter("reference");const c=n.safe(n.associationId(e),{before:l,after:"]",...s.current()});return o(),n.stack=d,a(),i==="full"||!u||u!==c?l+=s.move(c+"]"):i==="shortcut"?l=l.slice(0,-1):l+=s.move("]"),l}function GC(){return"!"}Qg.peek=ZC;function Qg(e,t,n){let r=e.value||"",i="`",a=-1;for(;new RegExp("(^|[^`])"+i+"([^`]|$)").test(r);)i+="`";for(/[^ \r\n]/.test(r)&&(/^[ \r\n]/.test(r)&&/[ \r\n]$/.test(r)||/^`|`$/.test(r))&&(r=" "+r+" ");++a<n.unsafe.length;){const o=n.unsafe[a],s=n.compilePattern(o);let l;if(o.atBreak)for(;l=s.exec(r);){let u=l.index;r.charCodeAt(u)===10&&r.charCodeAt(u-1)===13&&u--,r=r.slice(0,u)+" "+r.slice(l.index+1)}}return i+r+i}function ZC(){return"`"}function Kg(e,t){const n=xc(e);return!!(!t.options.resourceLink&&e.url&&!e.title&&e.children&&e.children.length===1&&e.children[0].type==="text"&&(n===e.url||"mailto:"+n===e.url)&&/^[a-z][a-z+.-]+:/i.test(e.url)&&!/[\0- <>\u007F]/.test(e.url))}Xg.peek=JC;function Xg(e,t,n,r){const i=Tc(n),a=i==='"'?"Quote":"Apostrophe",o=n.createTracker(r);let s,l;if(Kg(e,n)){const d=n.stack;n.stack=[],s=n.enter("autolink");let c=o.move("<");return c+=o.move(n.containerPhrasing(e,{before:c,after:">",...o.current()})),c+=o.move(">"),s(),n.stack=d,c}s=n.enter("link"),l=n.enter("label");let u=o.move("[");return u+=o.move(n.containerPhrasing(e,{before:u,after:"](",...o.current()})),u+=o.move("]("),l(),!e.url&&e.title||/[\0- \u007F]/.test(e.url)?(l=n.enter("destinationLiteral"),u+=o.move("<"),u+=o.move(n.safe(e.url,{before:u,after:">",...o.current()})),u+=o.move(">")):(l=n.enter("destinationRaw"),u+=o.move(n.safe(e.url,{before:u,after:e.title?" ":")",...o.current()}))),l(),e.title&&(l=n.enter(`title${a}`),u+=o.move(" "+i),u+=o.move(n.safe(e.title,{before:u,after:i,...o.current()})),u+=o.move(i),l()),u+=o.move(")"),s(),u}function JC(e,t,n){return Kg(e,n)?"<":"["}Yg.peek=eE;function Yg(e,t,n,r){const i=e.referenceType,a=n.enter("linkReference");let o=n.enter("label");const s=n.createTracker(r);let l=s.move("[");const u=n.containerPhrasing(e,{before:l,after:"]",...s.current()});l+=s.move(u+"]["),o();const d=n.stack;n.stack=[],o=n.enter("reference");const c=n.safe(n.associationId(e),{before:l,after:"]",...s.current()});return o(),n.stack=d,a(),i==="full"||!u||u!==c?l+=s.move(c+"]"):i==="shortcut"?l=l.slice(0,-1):l+=s.move("]"),l}function eE(){return"["}function Pc(e){const t=e.options.bullet||"*";if(t!=="*"&&t!=="+"&&t!=="-")throw new Error("Cannot serialize items with `"+t+"` for `options.bullet`, expected `*`, `+`, or `-`");return t}function tE(e){const t=Pc(e),n=e.options.bulletOther;if(!n)return t==="*"?"-":"*";if(n!=="*"&&n!=="+"&&n!=="-")throw new Error("Cannot serialize items with `"+n+"` for `options.bulletOther`, expected `*`, `+`, or `-`");if(n===t)throw new Error("Expected `bullet` (`"+t+"`) and `bulletOther` (`"+n+"`) to be different");return n}function nE(e){const t=e.options.bulletOrdered||".";if(t!=="."&&t!==")")throw new Error("Cannot serialize items with `"+t+"` for `options.bulletOrdered`, expected `.` or `)`");return t}function Gg(e){const t=e.options.rule||"*";if(t!=="*"&&t!=="-"&&t!=="_")throw new Error("Cannot serialize rules with `"+t+"` for `options.rule`, expected `*`, `-`, or `_`");return t}function rE(e,t,n,r){const i=n.enter("list"),a=n.bulletCurrent;let o=e.ordered?nE(n):Pc(n);const s=e.ordered?o==="."?")":".":tE(n);let l=t&&n.bulletLastUsed?o===n.bulletLastUsed:!1;if(!e.ordered){const d=e.children?e.children[0]:void 0;if((o==="*"||o==="-")&&d&&(!d.children||!d.children[0])&&n.stack[n.stack.length-1]==="list"&&n.stack[n.stack.length-2]==="listItem"&&n.stack[n.stack.length-3]==="list"&&n.stack[n.stack.length-4]==="listItem"&&n.indexStack[n.indexStack.length-1]===0&&n.indexStack[n.indexStack.length-2]===0&&n.indexStack[n.indexStack.length-3]===0&&(l=!0),Gg(n)===o&&d){let c=-1;for(;++c<e.children.length;){const p=e.children[c];if(p&&p.type==="listItem"&&p.children&&p.children[0]&&p.children[0].type==="thematicBreak"){l=!0;break}}}}l&&(o=s),n.bulletCurrent=o;const u=n.containerFlow(e,r);return n.bulletLastUsed=o,n.bulletCurrent=a,i(),u}function iE(e){const t=e.options.listItemIndent||"one";if(t!=="tab"&&t!=="one"&&t!=="mixed")throw new Error("Cannot serialize items with `"+t+"` for `options.listItemIndent`, expected `tab`, `one`, or `mixed`");return t}function aE(e,t,n,r){const i=iE(n);let a=n.bulletCurrent||Pc(n);t&&t.type==="list"&&t.ordered&&(a=(typeof t.start=="number"&&t.start>-1?t.start:1)+(n.options.incrementListMarker===!1?0:t.children.indexOf(e))+a);let o=a.length+1;(i==="tab"||i==="mixed"&&(t&&t.type==="list"&&t.spread||e.spread))&&(o=Math.ceil(o/4)*4);const s=n.createTracker(r);s.move(a+" ".repeat(o-a.length)),s.shift(o);const l=n.enter("listItem"),u=n.indentLines(n.containerFlow(e,s.current()),d);return l(),u;function d(c,p,f){return p?(f?"":" ".repeat(o))+c:(f?a:a+" ".repeat(o-a.length))+c}}function oE(e,t,n,r){const i=n.enter("paragraph"),a=n.enter("phrasing"),o=n.containerPhrasing(e,r);return a(),i(),o}const sE=Zo(["break","delete","emphasis","footnote","footnoteReference","image","imageReference","inlineCode","inlineMath","link","linkReference","mdxJsxTextElement","mdxTextExpression","strong","text","textDirective"]);function lE(e,t,n,r){return(e.children.some(function(o){return sE(o)})?n.containerPhrasing:n.containerFlow).call(n,e,r)}function uE(e){const t=e.options.strong||"*";if(t!=="*"&&t!=="_")throw new Error("Cannot serialize strong with `"+t+"` for `options.strong`, expected `*`, or `_`");return t}Zg.peek=cE;function Zg(e,t,n,r){const i=uE(n),a=n.enter("strong"),o=n.createTracker(r),s=o.move(i+i);let l=o.move(n.containerPhrasing(e,{after:i,before:s,...o.current()}));const u=l.charCodeAt(0),d=zo(r.before.charCodeAt(r.before.length-1),u,i);d.inside&&(l=Ki(u)+l.slice(1));const c=l.charCodeAt(l.length-1),p=zo(r.after.charCodeAt(0),c,i);p.inside&&(l=l.slice(0,-1)+Ki(c));const f=o.move(i+i);return a(),n.attentionEncodeSurroundingInfo={after:p.outside,before:d.outside},s+l+f}function cE(e,t,n){return n.options.strong||"*"}function dE(e,t,n,r){return n.safe(e.value,r)}function pE(e){const t=e.options.ruleRepetition||3;if(t<3)throw new Error("Cannot serialize rules with repetition `"+t+"` for `options.ruleRepetition`, expected `3` or more");return t}function fE(e,t,n){const r=(Gg(n)+(n.options.ruleSpaces?" ":"")).repeat(pE(n));return n.options.ruleSpaces?r.slice(0,-1):r}const Jg={blockquote:OC,break:tf,code:BC,definition:VC,emphasis:Ug,hardBreak:tf,heading:KC,html:Vg,image:Hg,imageReference:$g,inlineCode:Qg,link:Xg,linkReference:Yg,list:rE,listItem:aE,paragraph:oE,root:lE,strong:Zg,text:dE,thematicBreak:fE};function mE(){return{enter:{table:hE,tableData:nf,tableHeader:nf,tableRow:yE},exit:{codeText:xE,table:gE,tableData:Zs,tableHeader:Zs,tableRow:Zs}}}function hE(e){const t=e._align;this.enter({type:"table",align:t.map(function(n){return n==="none"?null:n}),children:[]},e),this.data.inTable=!0}function gE(e){this.exit(e),this.data.inTable=void 0}function yE(e){this.enter({type:"tableRow",children:[]},e)}function Zs(e){this.exit(e)}function nf(e){this.enter({type:"tableCell",children:[]},e)}function xE(e){let t=this.resume();this.data.inTable&&(t=t.replace(/\\([\\|])/g,vE));const n=this.stack[this.stack.length-1];n.type,n.value=t,this.exit(e)}function vE(e,t){return t==="|"?t:e}function _E(e){const t=e||{},n=t.tableCellPadding,r=t.tablePipeAlign,i=t.stringLength,a=n?" ":"|";return{unsafe:[{character:"\r",inConstruct:"tableCell"},{character:`
`,inConstruct:"tableCell"},{atBreak:!0,character:"|",after:"[	 :-]"},{character:"|",inConstruct:"tableCell"},{atBreak:!0,character:":",after:"-"},{atBreak:!0,character:"-",after:"[:|-]"}],handlers:{inlineCode:p,table:o,tableCell:l,tableRow:s}};function o(f,y,v,k){return u(d(f,v,k),f.align)}function s(f,y,v,k){const m=c(f,v,k),g=u([m]);return g.slice(0,g.indexOf(`
`))}function l(f,y,v,k){const m=v.enter("tableCell"),g=v.enter("phrasing"),x=v.containerPhrasing(f,{...k,before:a,after:a});return g(),m(),x}function u(f,y){return RC(f,{align:y,alignDelimiters:r,padding:n,stringLength:i})}function d(f,y,v){const k=f.children;let m=-1;const g=[],x=y.enter("table");for(;++m<k.length;)g[m]=c(k[m],y,v);return x(),g}function c(f,y,v){const k=f.children;let m=-1;const g=[],x=y.enter("tableRow");for(;++m<k.length;)g[m]=l(k[m],f,y,v);return x(),g}function p(f,y,v){let k=Jg.inlineCode(f,y,v);return v.stack.includes("tableCell")&&(k=k.replace(/\|/g,"\\$&")),k}}function wE(){return{exit:{taskListCheckValueChecked:rf,taskListCheckValueUnchecked:rf,paragraph:kE}}}function bE(){return{unsafe:[{atBreak:!0,character:"-",after:"[:|-]"}],handlers:{listItem:SE}}}function rf(e){const t=this.stack[this.stack.length-2];t.type,t.checked=e.type==="taskListCheckValueChecked"}function kE(e){const t=this.stack[this.stack.length-2];if(t&&t.type==="listItem"&&typeof t.checked=="boolean"){const n=this.stack[this.stack.length-1];n.type;const r=n.children[0];if(r&&r.type==="text"){const i=t.children;let a=-1,o;for(;++a<i.length;){const s=i[a];if(s.type==="paragraph"){o=s;break}}o===n&&(r.value=r.value.slice(1),r.value.length===0?n.children.shift():n.position&&r.position&&typeof r.position.start.offset=="number"&&(r.position.start.column++,r.position.start.offset++,n.position.start=Object.assign({},r.position.start)))}}this.exit(e)}function SE(e,t,n,r){const i=e.children[0],a=typeof e.checked=="boolean"&&i&&i.type==="paragraph",o="["+(e.checked?"x":" ")+"] ",s=n.createTracker(r);a&&s.move(o);let l=Jg.listItem(e,t,n,{...r,...s.current()});return a&&(l=l.replace(/^(?:[*+-]|\d+\.)([\r\n]| {1,3})/,u)),l;function u(d){return d+o}}function CE(){return[rC(),SC(),TC(),mE(),wE()]}function EE(e){return{extensions:[iC(),CC(e),PC(),_E(e),bE()]}}const zE={tokenize:jE,partial:!0},ey={tokenize:RE,partial:!0},ty={tokenize:IE,partial:!0},ny={tokenize:OE,partial:!0},TE={tokenize:DE,partial:!0},ry={name:"wwwAutolink",tokenize:AE,previous:ay},iy={name:"protocolAutolink",tokenize:LE,previous:oy},an={name:"emailAutolink",tokenize:NE,previous:sy},Ut={};function PE(){return{text:Ut}}let On=48;for(;On<123;)Ut[On]=an,On++,On===58?On=65:On===91&&(On=97);Ut[43]=an;Ut[45]=an;Ut[46]=an;Ut[95]=an;Ut[72]=[an,iy];Ut[104]=[an,iy];Ut[87]=[an,ry];Ut[119]=[an,ry];function NE(e,t,n){const r=this;let i,a;return o;function o(c){return!mu(c)||!sy.call(r,r.previous)||Nc(r.events)?n(c):(e.enter("literalAutolink"),e.enter("literalAutolinkEmail"),s(c))}function s(c){return mu(c)?(e.consume(c),s):c===64?(e.consume(c),l):n(c)}function l(c){return c===46?e.check(TE,d,u)(c):c===45||c===95||Fe(c)?(a=!0,e.consume(c),l):d(c)}function u(c){return e.consume(c),i=!0,l}function d(c){return a&&i&&Ve(r.previous)?(e.exit("literalAutolinkEmail"),e.exit("literalAutolink"),t(c)):n(c)}}function AE(e,t,n){const r=this;return i;function i(o){return o!==87&&o!==119||!ay.call(r,r.previous)||Nc(r.events)?n(o):(e.enter("literalAutolink"),e.enter("literalAutolinkWww"),e.check(zE,e.attempt(ey,e.attempt(ty,a),n),n)(o))}function a(o){return e.exit("literalAutolinkWww"),e.exit("literalAutolink"),t(o)}}function LE(e,t,n){const r=this;let i="",a=!1;return o;function o(c){return(c===72||c===104)&&oy.call(r,r.previous)&&!Nc(r.events)?(e.enter("literalAutolink"),e.enter("literalAutolinkHttp"),i+=String.fromCodePoint(c),e.consume(c),s):n(c)}function s(c){if(Ve(c)&&i.length<5)return i+=String.fromCodePoint(c),e.consume(c),s;if(c===58){const p=i.toLowerCase();if(p==="http"||p==="https")return e.consume(c),l}return n(c)}function l(c){return c===47?(e.consume(c),a?u:(a=!0,l)):n(c)}function u(c){return c===null||ko(c)||ce(c)||Gn(c)||Xo(c)?n(c):e.attempt(ey,e.attempt(ty,d),n)(c)}function d(c){return e.exit("literalAutolinkHttp"),e.exit("literalAutolink"),t(c)}}function jE(e,t,n){let r=0;return i;function i(o){return(o===87||o===119)&&r<3?(r++,e.consume(o),i):o===46&&r===3?(e.consume(o),a):n(o)}function a(o){return o===null?n(o):t(o)}}function RE(e,t,n){let r,i,a;return o;function o(u){return u===46||u===95?e.check(ny,l,s)(u):u===null||ce(u)||Gn(u)||u!==45&&Xo(u)?l(u):(a=!0,e.consume(u),o)}function s(u){return u===95?r=!0:(i=r,r=void 0),e.consume(u),o}function l(u){return i||r||!a?n(u):t(u)}}function IE(e,t){let n=0,r=0;return i;function i(o){return o===40?(n++,e.consume(o),i):o===41&&r<n?a(o):o===33||o===34||o===38||o===39||o===41||o===42||o===44||o===46||o===58||o===59||o===60||o===63||o===93||o===95||o===126?e.check(ny,t,a)(o):o===null||ce(o)||Gn(o)?t(o):(e.consume(o),i)}function a(o){return o===41&&r++,e.consume(o),i}}function OE(e,t,n){return r;function r(s){return s===33||s===34||s===39||s===41||s===42||s===44||s===46||s===58||s===59||s===63||s===95||s===126?(e.consume(s),r):s===38?(e.consume(s),a):s===93?(e.consume(s),i):s===60||s===null||ce(s)||Gn(s)?t(s):n(s)}function i(s){return s===null||s===40||s===91||ce(s)||Gn(s)?t(s):r(s)}function a(s){return Ve(s)?o(s):n(s)}function o(s){return s===59?(e.consume(s),r):Ve(s)?(e.consume(s),o):n(s)}}function DE(e,t,n){return r;function r(a){return e.consume(a),i}function i(a){return Fe(a)?n(a):t(a)}}function ay(e){return e===null||e===40||e===42||e===95||e===91||e===93||e===126||ce(e)}function oy(e){return!Ve(e)}function sy(e){return!(e===47||mu(e))}function mu(e){return e===43||e===45||e===46||e===95||Fe(e)}function Nc(e){let t=e.length,n=!1;for(;t--;){const r=e[t][1];if((r.type==="labelLink"||r.type==="labelImage")&&!r._balanced){n=!0;break}if(r._gfmAutolinkLiteralWalkedInto){n=!1;break}}return e.length>0&&!n&&(e[e.length-1][1]._gfmAutolinkLiteralWalkedInto=!0),n}const ME={tokenize:$E,partial:!0};function FE(){return{document:{91:{name:"gfmFootnoteDefinition",tokenize:UE,continuation:{tokenize:VE},exit:HE}},text:{91:{name:"gfmFootnoteCall",tokenize:BE},93:{name:"gfmPotentialFootnoteCall",add:"after",tokenize:qE,resolveTo:WE}}}}function qE(e,t,n){const r=this;let i=r.events.length;const a=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let o;for(;i--;){const l=r.events[i][1];if(l.type==="labelImage"){o=l;break}if(l.type==="gfmFootnoteCall"||l.type==="labelLink"||l.type==="label"||l.type==="image"||l.type==="link")break}return s;function s(l){if(!o||!o._balanced)return n(l);const u=Rt(r.sliceSerialize({start:o.end,end:r.now()}));return u.codePointAt(0)!==94||!a.includes(u.slice(1))?n(l):(e.enter("gfmFootnoteCallLabelMarker"),e.consume(l),e.exit("gfmFootnoteCallLabelMarker"),t(l))}}function WE(e,t){let n=e.length;for(;n--;)if(e[n][1].type==="labelImage"&&e[n][0]==="enter"){e[n][1];break}e[n+1][1].type="data",e[n+3][1].type="gfmFootnoteCallLabelMarker";const r={type:"gfmFootnoteCall",start:Object.assign({},e[n+3][1].start),end:Object.assign({},e[e.length-1][1].end)},i={type:"gfmFootnoteCallMarker",start:Object.assign({},e[n+3][1].end),end:Object.assign({},e[n+3][1].end)};i.end.column++,i.end.offset++,i.end._bufferIndex++;const a={type:"gfmFootnoteCallString",start:Object.assign({},i.end),end:Object.assign({},e[e.length-1][1].start)},o={type:"chunkString",contentType:"string",start:Object.assign({},a.start),end:Object.assign({},a.end)},s=[e[n+1],e[n+2],["enter",r,t],e[n+3],e[n+4],["enter",i,t],["exit",i,t],["enter",a,t],["enter",o,t],["exit",o,t],["exit",a,t],e[e.length-2],e[e.length-1],["exit",r,t]];return e.splice(n,e.length-n+1,...s),e}function BE(e,t,n){const r=this,i=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let a=0,o;return s;function s(c){return e.enter("gfmFootnoteCall"),e.enter("gfmFootnoteCallLabelMarker"),e.consume(c),e.exit("gfmFootnoteCallLabelMarker"),l}function l(c){return c!==94?n(c):(e.enter("gfmFootnoteCallMarker"),e.consume(c),e.exit("gfmFootnoteCallMarker"),e.enter("gfmFootnoteCallString"),e.enter("chunkString").contentType="string",u)}function u(c){if(a>999||c===93&&!o||c===null||c===91||ce(c))return n(c);if(c===93){e.exit("chunkString");const p=e.exit("gfmFootnoteCallString");return i.includes(Rt(r.sliceSerialize(p)))?(e.enter("gfmFootnoteCallLabelMarker"),e.consume(c),e.exit("gfmFootnoteCallLabelMarker"),e.exit("gfmFootnoteCall"),t):n(c)}return ce(c)||(o=!0),a++,e.consume(c),c===92?d:u}function d(c){return c===91||c===92||c===93?(e.consume(c),a++,u):u(c)}}function UE(e,t,n){const r=this,i=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let a,o=0,s;return l;function l(y){return e.enter("gfmFootnoteDefinition")._container=!0,e.enter("gfmFootnoteDefinitionLabel"),e.enter("gfmFootnoteDefinitionLabelMarker"),e.consume(y),e.exit("gfmFootnoteDefinitionLabelMarker"),u}function u(y){return y===94?(e.enter("gfmFootnoteDefinitionMarker"),e.consume(y),e.exit("gfmFootnoteDefinitionMarker"),e.enter("gfmFootnoteDefinitionLabelString"),e.enter("chunkString").contentType="string",d):n(y)}function d(y){if(o>999||y===93&&!s||y===null||y===91||ce(y))return n(y);if(y===93){e.exit("chunkString");const v=e.exit("gfmFootnoteDefinitionLabelString");return a=Rt(r.sliceSerialize(v)),e.enter("gfmFootnoteDefinitionLabelMarker"),e.consume(y),e.exit("gfmFootnoteDefinitionLabelMarker"),e.exit("gfmFootnoteDefinitionLabel"),p}return ce(y)||(s=!0),o++,e.consume(y),y===92?c:d}function c(y){return y===91||y===92||y===93?(e.consume(y),o++,d):d(y)}function p(y){return y===58?(e.enter("definitionMarker"),e.consume(y),e.exit("definitionMarker"),i.includes(a)||i.push(a),re(e,f,"gfmFootnoteDefinitionWhitespace")):n(y)}function f(y){return t(y)}}function VE(e,t,n){return e.check(ia,t,e.attempt(ME,t,n))}function HE(e){e.exit("gfmFootnoteDefinition")}function $E(e,t,n){const r=this;return re(e,i,"gfmFootnoteDefinitionIndent",5);function i(a){const o=r.events[r.events.length-1];return o&&o[1].type==="gfmFootnoteDefinitionIndent"&&o[2].sliceSerialize(o[1],!0).length===4?t(a):n(a)}}function QE(e){let n=(e||{}).singleTilde;const r={name:"strikethrough",tokenize:a,resolveAll:i};return n==null&&(n=!0),{text:{126:r},insideSpan:{null:[r]},attentionMarkers:{null:[126]}};function i(o,s){let l=-1;for(;++l<o.length;)if(o[l][0]==="enter"&&o[l][1].type==="strikethroughSequenceTemporary"&&o[l][1]._close){let u=l;for(;u--;)if(o[u][0]==="exit"&&o[u][1].type==="strikethroughSequenceTemporary"&&o[u][1]._open&&o[l][1].end.offset-o[l][1].start.offset===o[u][1].end.offset-o[u][1].start.offset){o[l][1].type="strikethroughSequence",o[u][1].type="strikethroughSequence";const d={type:"strikethrough",start:Object.assign({},o[u][1].start),end:Object.assign({},o[l][1].end)},c={type:"strikethroughText",start:Object.assign({},o[u][1].end),end:Object.assign({},o[l][1].start)},p=[["enter",d,s],["enter",o[u][1],s],["exit",o[u][1],s],["enter",c,s]],f=s.parser.constructs.insideSpan.null;f&&st(p,p.length,0,Yo(f,o.slice(u+1,l),s)),st(p,p.length,0,[["exit",c,s],["enter",o[l][1],s],["exit",o[l][1],s],["exit",d,s]]),st(o,u-1,l-u+3,p),l=u+p.length-2;break}}for(l=-1;++l<o.length;)o[l][1].type==="strikethroughSequenceTemporary"&&(o[l][1].type="data");return o}function a(o,s,l){const u=this.previous,d=this.events;let c=0;return p;function p(y){return u===126&&d[d.length-1][1].type!=="characterEscape"?l(y):(o.enter("strikethroughSequenceTemporary"),f(y))}function f(y){const v=Fr(u);if(y===126)return c>1?l(y):(o.consume(y),c++,f);if(c<2&&!n)return l(y);const k=o.exit("strikethroughSequenceTemporary"),m=Fr(y);return k._open=!m||m===2&&!!v,k._close=!v||v===2&&!!m,s(y)}}}class KE{constructor(){this.map=[]}add(t,n,r){XE(this,t,n,r)}consume(t){if(this.map.sort(function(a,o){return a[0]-o[0]}),this.map.length===0)return;let n=this.map.length;const r=[];for(;n>0;)n-=1,r.push(t.slice(this.map[n][0]+this.map[n][1]),this.map[n][2]),t.length=this.map[n][0];r.push(t.slice()),t.length=0;let i=r.pop();for(;i;){for(const a of i)t.push(a);i=r.pop()}this.map.length=0}}function XE(e,t,n,r){let i=0;if(!(n===0&&r.length===0)){for(;i<e.map.length;){if(e.map[i][0]===t){e.map[i][1]+=n,e.map[i][2].push(...r);return}i+=1}e.map.push([t,n,r])}}function YE(e,t){let n=!1;const r=[];for(;t<e.length;){const i=e[t];if(n){if(i[0]==="enter")i[1].type==="tableContent"&&r.push(e[t+1][1].type==="tableDelimiterMarker"?"left":"none");else if(i[1].type==="tableContent"){if(e[t-1][1].type==="tableDelimiterMarker"){const a=r.length-1;r[a]=r[a]==="left"?"center":"right"}}else if(i[1].type==="tableDelimiterRow")break}else i[0]==="enter"&&i[1].type==="tableDelimiterRow"&&(n=!0);t+=1}return r}function GE(){return{flow:{null:{name:"table",tokenize:ZE,resolveAll:JE}}}}function ZE(e,t,n){const r=this;let i=0,a=0,o;return s;function s(b){let I=r.events.length-1;for(;I>-1;){const ee=r.events[I][1].type;if(ee==="lineEnding"||ee==="linePrefix")I--;else break}const W=I>-1?r.events[I][1].type:null,X=W==="tableHead"||W==="tableRow"?S:l;return X===S&&r.parser.lazy[r.now().line]?n(b):X(b)}function l(b){return e.enter("tableHead"),e.enter("tableRow"),u(b)}function u(b){return b===124||(o=!0,a+=1),d(b)}function d(b){return b===null?n(b):K(b)?a>1?(a=0,r.interrupt=!0,e.exit("tableRow"),e.enter("lineEnding"),e.consume(b),e.exit("lineEnding"),f):n(b):Z(b)?re(e,d,"whitespace")(b):(a+=1,o&&(o=!1,i+=1),b===124?(e.enter("tableCellDivider"),e.consume(b),e.exit("tableCellDivider"),o=!0,d):(e.enter("data"),c(b)))}function c(b){return b===null||b===124||ce(b)?(e.exit("data"),d(b)):(e.consume(b),b===92?p:c)}function p(b){return b===92||b===124?(e.consume(b),c):c(b)}function f(b){return r.interrupt=!1,r.parser.lazy[r.now().line]?n(b):(e.enter("tableDelimiterRow"),o=!1,Z(b)?re(e,y,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(b):y(b))}function y(b){return b===45||b===58?k(b):b===124?(o=!0,e.enter("tableCellDivider"),e.consume(b),e.exit("tableCellDivider"),v):z(b)}function v(b){return Z(b)?re(e,k,"whitespace")(b):k(b)}function k(b){return b===58?(a+=1,o=!0,e.enter("tableDelimiterMarker"),e.consume(b),e.exit("tableDelimiterMarker"),m):b===45?(a+=1,m(b)):b===null||K(b)?C(b):z(b)}function m(b){return b===45?(e.enter("tableDelimiterFiller"),g(b)):z(b)}function g(b){return b===45?(e.consume(b),g):b===58?(o=!0,e.exit("tableDelimiterFiller"),e.enter("tableDelimiterMarker"),e.consume(b),e.exit("tableDelimiterMarker"),x):(e.exit("tableDelimiterFiller"),x(b))}function x(b){return Z(b)?re(e,C,"whitespace")(b):C(b)}function C(b){return b===124?y(b):b===null||K(b)?!o||i!==a?z(b):(e.exit("tableDelimiterRow"),e.exit("tableHead"),t(b)):z(b)}function z(b){return n(b)}function S(b){return e.enter("tableRow"),T(b)}function T(b){return b===124?(e.enter("tableCellDivider"),e.consume(b),e.exit("tableCellDivider"),T):b===null||K(b)?(e.exit("tableRow"),t(b)):Z(b)?re(e,T,"whitespace")(b):(e.enter("data"),N(b))}function N(b){return b===null||b===124||ce(b)?(e.exit("data"),T(b)):(e.consume(b),b===92?F:N)}function F(b){return b===92||b===124?(e.consume(b),N):N(b)}}function JE(e,t){let n=-1,r=!0,i=0,a=[0,0,0,0],o=[0,0,0,0],s=!1,l=0,u,d,c;const p=new KE;for(;++n<e.length;){const f=e[n],y=f[1];f[0]==="enter"?y.type==="tableHead"?(s=!1,l!==0&&(af(p,t,l,u,d),d=void 0,l=0),u={type:"table",start:Object.assign({},y.start),end:Object.assign({},y.end)},p.add(n,0,[["enter",u,t]])):y.type==="tableRow"||y.type==="tableDelimiterRow"?(r=!0,c=void 0,a=[0,0,0,0],o=[0,n+1,0,0],s&&(s=!1,d={type:"tableBody",start:Object.assign({},y.start),end:Object.assign({},y.end)},p.add(n,0,[["enter",d,t]])),i=y.type==="tableDelimiterRow"?2:d?3:1):i&&(y.type==="data"||y.type==="tableDelimiterMarker"||y.type==="tableDelimiterFiller")?(r=!1,o[2]===0&&(a[1]!==0&&(o[0]=o[1],c=La(p,t,a,i,void 0,c),a=[0,0,0,0]),o[2]=n)):y.type==="tableCellDivider"&&(r?r=!1:(a[1]!==0&&(o[0]=o[1],c=La(p,t,a,i,void 0,c)),a=o,o=[a[1],n,0,0])):y.type==="tableHead"?(s=!0,l=n):y.type==="tableRow"||y.type==="tableDelimiterRow"?(l=n,a[1]!==0?(o[0]=o[1],c=La(p,t,a,i,n,c)):o[1]!==0&&(c=La(p,t,o,i,n,c)),i=0):i&&(y.type==="data"||y.type==="tableDelimiterMarker"||y.type==="tableDelimiterFiller")&&(o[3]=n)}for(l!==0&&af(p,t,l,u,d),p.consume(t.events),n=-1;++n<t.events.length;){const f=t.events[n];f[0]==="enter"&&f[1].type==="table"&&(f[1]._align=YE(t.events,n))}return e}function La(e,t,n,r,i,a){const o=r===1?"tableHeader":r===2?"tableDelimiter":"tableData",s="tableContent";n[0]!==0&&(a.end=Object.assign({},or(t.events,n[0])),e.add(n[0],0,[["exit",a,t]]));const l=or(t.events,n[1]);if(a={type:o,start:Object.assign({},l),end:Object.assign({},l)},e.add(n[1],0,[["enter",a,t]]),n[2]!==0){const u=or(t.events,n[2]),d=or(t.events,n[3]),c={type:s,start:Object.assign({},u),end:Object.assign({},d)};if(e.add(n[2],0,[["enter",c,t]]),r!==2){const p=t.events[n[2]],f=t.events[n[3]];if(p[1].end=Object.assign({},f[1].end),p[1].type="chunkText",p[1].contentType="text",n[3]>n[2]+1){const y=n[2]+1,v=n[3]-n[2]-1;e.add(y,v,[])}}e.add(n[3]+1,0,[["exit",c,t]])}return i!==void 0&&(a.end=Object.assign({},or(t.events,i)),e.add(i,0,[["exit",a,t]]),a=void 0),a}function af(e,t,n,r,i){const a=[],o=or(t.events,n);i&&(i.end=Object.assign({},o),a.push(["exit",i,t])),r.end=Object.assign({},o),a.push(["exit",r,t]),e.add(n+1,0,a)}function or(e,t){const n=e[t],r=n[0]==="enter"?"start":"end";return n[1][r]}const ez={name:"tasklistCheck",tokenize:nz};function tz(){return{text:{91:ez}}}function nz(e,t,n){const r=this;return i;function i(l){return r.previous!==null||!r._gfmTasklistFirstContentOfListItem?n(l):(e.enter("taskListCheck"),e.enter("taskListCheckMarker"),e.consume(l),e.exit("taskListCheckMarker"),a)}function a(l){return ce(l)?(e.enter("taskListCheckValueUnchecked"),e.consume(l),e.exit("taskListCheckValueUnchecked"),o):l===88||l===120?(e.enter("taskListCheckValueChecked"),e.consume(l),e.exit("taskListCheckValueChecked"),o):n(l)}function o(l){return l===93?(e.enter("taskListCheckMarker"),e.consume(l),e.exit("taskListCheckMarker"),e.exit("taskListCheck"),s):n(l)}function s(l){return K(l)?t(l):Z(l)?e.check({tokenize:rz},t,n)(l):n(l)}}function rz(e,t,n){return re(e,r,"whitespace");function r(i){return i===null?n(i):t(i)}}function iz(e){return ng([PE(),FE(),QE(e),GE(),tz()])}const az={};function oz(e){const t=this,n=e||az,r=t.data(),i=r.micromarkExtensions||(r.micromarkExtensions=[]),a=r.fromMarkdownExtensions||(r.fromMarkdownExtensions=[]),o=r.toMarkdownExtensions||(r.toMarkdownExtensions=[]);i.push(iz(n)),a.push(CE()),o.push(EE(n))}function sz({problem:e}){return h.jsxs("div",{className:"prose prose-sm max-w-none overflow-hidden",children:[h.jsxs("div",{className:"flex items-center gap-3 mb-4",children:[h.jsx("h1",{className:"text-xl font-bold text-gray-900 m-0",children:e.title}),h.jsx("span",{className:`px-2 py-0.5 rounded text-xs font-medium ${e.difficulty==="easy"?"bg-green-100 text-green-700":e.difficulty==="medium"?"bg-yellow-100 text-yellow-700":"bg-red-100 text-red-700"}`,children:e.difficulty})]}),h.jsx(Sg,{remarkPlugins:[oz],components:{h2:({children:t})=>h.jsx("h2",{className:"text-lg font-semibold text-gray-900 mt-6 mb-3",children:t}),h3:({children:t})=>h.jsx("h3",{className:"text-base font-medium text-gray-700 mt-4 mb-2",children:t}),p:({children:t})=>h.jsx("p",{className:"text-gray-600 mb-3 leading-relaxed",children:t}),code:({className:t,children:n})=>(t==null?void 0:t.includes("language-"))?h.jsx("pre",{className:"bg-gray-100 rounded-lg p-4 overflow-x-auto my-3",children:h.jsx("code",{className:"text-sm text-gray-800",children:n})}):h.jsx("code",{className:"bg-gray-100 px-1.5 py-0.5 rounded text-primary-600 text-sm break-words",children:n}),ul:({children:t})=>h.jsx("ul",{className:"list-disc list-inside text-gray-600 space-y-1 mb-3",children:t}),li:({children:t})=>h.jsx("li",{className:"text-gray-600",children:t}),table:({children:t})=>h.jsx("div",{className:"overflow-x-auto my-4",children:h.jsx("table",{className:"min-w-full border-collapse border border-gray-300 text-sm",children:t})}),thead:({children:t})=>h.jsx("thead",{className:"bg-gray-100",children:t}),tbody:({children:t})=>h.jsx("tbody",{className:"divide-y divide-gray-200",children:t}),tr:({children:t})=>h.jsx("tr",{children:t}),th:({children:t})=>h.jsx("th",{className:"border border-gray-300 px-3 py-2 text-left font-semibold text-gray-700",children:t}),td:({children:t})=>h.jsx("td",{className:"border border-gray-300 px-3 py-2 text-gray-600",children:t})},children:e.description})]})}function lz({examples:e}){return h.jsxs("div",{className:"mt-6",children:[h.jsx("h3",{className:"text-base font-medium text-gray-900 mb-3",children:"Examples"}),h.jsx("div",{className:"space-y-4",children:e.map((t,n)=>h.jsx("div",{className:"bg-white rounded-lg p-4 border border-gray-200 shadow-sm",children:h.jsxs("div",{className:"space-y-2",children:[h.jsxs("div",{className:"overflow-hidden",children:[h.jsx("span",{className:"text-gray-500 text-sm",children:"Input:"}),h.jsx("pre",{className:"mt-1 text-sm text-primary-600 font-mono whitespace-pre-wrap break-all",children:t.input})]}),h.jsxs("div",{className:"overflow-hidden",children:[h.jsx("span",{className:"text-gray-500 text-sm",children:"Output:"}),h.jsx("pre",{className:"mt-1 text-sm text-green-600 font-mono whitespace-pre-wrap break-all",children:t.output})]}),t.explanation&&h.jsxs("div",{className:"pt-2 border-t border-gray-200",children:[h.jsx("span",{className:"text-gray-500 text-sm",children:"Explanation: "}),h.jsx("span",{className:"text-gray-600 text-sm",children:t.explanation})]})]})},n))})]})}function uz({hints:e,solution:t}){const[n,r]=E.useState(0),[i,a]=E.useState(!1),o=()=>{n<e.length&&r(l=>l+1)},s=()=>{a(!0)};return h.jsxs("div",{className:"mt-6 space-y-4",children:[h.jsxs("div",{children:[h.jsxs("div",{className:"flex items-center justify-between mb-3",children:[h.jsx("h3",{className:"text-base font-medium text-gray-900",children:"Hints"}),n<e.length&&h.jsxs("button",{onClick:o,className:"text-sm text-primary-600 hover:text-primary-500 transition-colors",children:["Show Hint (",n,"/",e.length,")"]})]}),n>0?h.jsx("div",{className:"space-y-2",children:e.slice(0,n).map((l,u)=>h.jsx("div",{className:"bg-yellow-50 border border-yellow-200 rounded-lg p-3",children:h.jsxs("div",{className:"flex items-start gap-2",children:[h.jsx("span",{className:"text-yellow-600 flex-shrink-0",children:"💡"}),h.jsx("span",{className:"text-gray-700 text-sm break-words overflow-hidden",children:l})]})},u))}):h.jsx("div",{className:"text-gray-400 text-sm italic",children:'Click "Show Hint" if you need help'})]}),h.jsxs("div",{children:[h.jsxs("div",{className:"flex items-center justify-between mb-3",children:[h.jsx("h3",{className:"text-base font-medium text-gray-900",children:"Solution"}),!i&&n>=e.length&&h.jsx("button",{onClick:s,className:"text-sm text-primary-600 hover:text-primary-500 transition-colors",children:"Reveal Solution"})]}),i?h.jsxs("div",{className:"bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm",children:[h.jsx("div",{className:"px-4 py-2 bg-gray-100 border-b border-gray-200",children:h.jsx("span",{className:"text-sm text-gray-600",children:"Solution Code"})}),h.jsx("pre",{className:"p-4 text-sm text-gray-300 font-mono overflow-x-auto bg-gray-900",children:t})]}):n<e.length?h.jsx("div",{className:"text-gray-400 text-sm italic",children:"Reveal all hints first to unlock the solution"}):h.jsx("div",{className:"text-gray-400 text-sm italic",children:'Click "Reveal Solution" to see the answer'})]})]})}function of({results:e,isRunning:t}){if(t)return h.jsxs("div",{className:"flex items-center gap-3 p-4 bg-white rounded-lg border border-gray-200 shadow-sm",children:[h.jsx("div",{className:"w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin"}),h.jsx("span",{className:"text-gray-600",children:"Running tests..."})]});if(e.length===0)return null;const n=e.filter(i=>i.passed).length,r=e.filter(i=>!i.hidden);return h.jsxs("div",{className:"bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm",children:[h.jsxs("div",{className:"px-4 py-3 border-b border-gray-200 flex items-center justify-between bg-gray-50",children:[h.jsx("span",{className:"font-medium text-gray-900",children:"Test Results"}),h.jsxs("span",{className:`text-sm font-medium ${n===e.length?"text-green-600":"text-yellow-600"}`,children:[n,"/",e.length," passed"]})]}),h.jsxs("div",{className:"divide-y divide-gray-200",children:[r.map(i=>h.jsxs("div",{className:"p-4",children:[h.jsxs("div",{className:"flex items-center gap-2 mb-2",children:[h.jsx("span",{className:`w-5 h-5 rounded-full flex items-center justify-center text-xs ${i.passed?"bg-green-100 text-green-600":"bg-red-100 text-red-600"}`,children:i.passed?"✓":"✗"}),h.jsxs("span",{className:"text-gray-700 text-sm font-medium",children:["Test ",i.id,": ",i.description]})]}),!i.passed&&h.jsxs("div",{className:"ml-7 space-y-1 text-sm",children:[h.jsxs("div",{children:[h.jsx("span",{className:"text-gray-500",children:"Expected: "}),h.jsx("span",{className:"text-green-600 font-mono",children:i.expected})]}),h.jsxs("div",{children:[h.jsx("span",{className:"text-gray-500",children:"Actual: "}),h.jsx("span",{className:"text-red-600 font-mono",children:i.actual})]})]})]},i.id)),e.some(i=>i.hidden)&&h.jsxs("div",{className:"p-4 text-gray-500 text-sm italic",children:["+ ",e.filter(i=>i.hidden).length," hidden test(s)",e.filter(i=>i.hidden&&i.passed).length===e.filter(i=>i.hidden).length?h.jsx("span",{className:"text-green-600 ml-2",children:"(all passed)"}):h.jsxs("span",{className:"text-red-600 ml-2",children:["(",e.filter(i=>i.hidden&&!i.passed).length," failed)"]})]})]})]})}function cz(){const{sectionId:e,problemId:t}=Ph(),{getProblemProgress:n,saveProblemCode:r,updateProblemStatus:i}=na(),{isLoading:a,isReady:o,output:s,runTests:l,clearOutput:u}=Ng(),d=t?P2(t):void 0,c=Mr.find(A=>A.id===e),[p,f]=E.useState(""),[y,v]=E.useState([]),[k,m]=E.useState(!1),[g,x]=E.useState(!1),[C,z]=E.useState([]),[S,T]=E.useState(0);E.useEffect(()=>{if(d&&e){const A=n(e,d.id);f(A.code||d.starterCode),z(d.testCases)}},[d,e,n]),E.useEffect(()=>{if(p&&d&&e&&p!==d.starterCode){const A=setTimeout(()=>{r(e,d.id,p)},1e3);return()=>clearTimeout(A)}},[p,d,e,r]);const N=E.useCallback(A=>{f(A)},[]),F=E.useCallback(()=>{d&&(f(d.starterCode),v([]),z(d.testCases),T(A=>A+1),u())},[d,u]),b=A=>{const $=[...A.matchAll(/def\s+(\w+)\s*\(/g)];return $.length>0?$[$.length-1][1]:"solution"},I=E.useCallback(async()=>{if(!(!d||!o||!e)){m(!0),v([]);try{const A=b(d.starterCode),$=await l(p,C,A);v($),$.every(O=>O.passed)&&i(e,d.id,"completed")}finally{m(!1)}}},[d,o,e,p,C,l,i]);if(!d||!c)return h.jsxs("div",{className:"text-center py-12",children:[h.jsx("h1",{className:"text-2xl font-bold text-gray-900 mb-4",children:"Problem Not Found"}),h.jsx(Nt,{to:"/",className:"text-primary-600 hover:text-primary-500",children:"Return to Home"})]});const W=c.problems.indexOf(d.id),X=c.problems[W+1],ee=d.description.replace(/[#*`]/g,"").split(`
`).filter(A=>A.trim()).slice(0,2).join(" ").substring(0,155);return h.jsxs("div",{className:"h-[calc(100vh-8rem)] flex flex-col -m-6",children:[h.jsx(Qo,{title:`${d.title} - ${c.title}`,description:`${ee}... Practice this ${d.difficulty} ML coding problem with instant feedback.`,canonical:`/problem/${e}/${t}`,keywords:`${d.title}, ${c.title}, machine learning, coding practice, ${d.difficulty}`}),h.jsxs("div",{className:"flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200",children:[h.jsxs("div",{className:"flex items-center gap-3",children:[h.jsxs(Nt,{to:`/section/${e}`,className:"text-gray-500 hover:text-gray-900 transition-colors",children:["← ",c.title]}),h.jsx("span",{className:"text-gray-300",children:"/"}),h.jsx("span",{className:"text-gray-900 font-medium",children:d.title})]}),h.jsxs("div",{className:"flex items-center gap-3",children:[!o&&h.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[h.jsx("div",{className:"w-2 h-2 bg-yellow-500 rounded-full animate-pulse"}),a?"Loading Python...":"Python ready"]}),o&&h.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[h.jsx("div",{className:"w-2 h-2 bg-green-500 rounded-full"}),"Python ready"]})]})]}),g?h.jsxs("div",{className:"flex flex-col flex-1 overflow-hidden bg-white",children:[h.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200",children:[h.jsxs("div",{className:"flex items-center gap-2",children:[h.jsx("button",{onClick:I,disabled:!o||k,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:k?"Running...":"Run Tests"}),h.jsx("span",{className:"text-xs text-gray-400",children:"or"}),h.jsxs("div",{className:"flex items-center gap-1 text-gray-500",children:[h.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm",children:"Shift"}),h.jsx("span",{className:"text-xs",children:"+"}),h.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm",children:"Enter"})]})]}),h.jsxs("div",{className:"flex items-center gap-2",children:[y.length>0&&y.every(A=>A.passed)&&X&&h.jsx(Nt,{to:`/problem/${e}/${X}`,className:"px-4 py-1.5 bg-green-100 text-green-700 text-sm font-medium rounded-md hover:bg-green-200 transition-colors",children:"Next Problem →"}),h.jsx("button",{onClick:()=>x(!g),className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",title:g?"Restore layout":"Maximize editor",children:g?"⊟ Restore":"⊞ Maximize"}),h.jsx("button",{onClick:F,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Reset"})]})]}),h.jsxs(Pr,{className:"flex flex-1 overflow-hidden",sizes:[60,40],minSize:200,gutterSize:8,gutterAlign:"center",children:[h.jsx("div",{className:"h-full overflow-hidden p-4",children:h.jsx(pu,{value:p,onChange:N,height:"100%",onRun:I})}),h.jsxs("div",{className:"overflow-auto p-4 space-y-4 bg-gray-50",children:[h.jsx(of,{results:y,isRunning:k}),h.jsx(fu,{output:s,isLoading:k,testCases:C,onTestCasesChange:z,problemId:t,resetKey:S})]})]})]}):h.jsxs(Pr,{className:"flex flex-1 overflow-hidden",sizes:[40,60],minSize:300,gutterSize:8,gutterAlign:"center",children:[h.jsxs("div",{className:"problem-panel overflow-y-auto overflow-x-hidden p-6 bg-gray-50 min-w-0",children:[h.jsx(sz,{problem:d}),h.jsx(lz,{examples:d.examples}),h.jsx(uz,{hints:d.hints,solution:d.solution})]}),h.jsxs("div",{className:"flex flex-col overflow-hidden bg-white",children:[h.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200",children:[h.jsxs("div",{className:"flex items-center gap-2",children:[h.jsx("button",{onClick:I,disabled:!o||k,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:k?"Running...":"Run Tests"}),h.jsx("span",{className:"text-xs text-gray-400",children:"or"}),h.jsxs("div",{className:"flex items-center gap-1 text-gray-500",children:[h.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm",children:"Shift"}),h.jsx("span",{className:"text-xs",children:"+"}),h.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm",children:"Enter"})]})]}),h.jsxs("div",{className:"flex items-center gap-2",children:[y.length>0&&y.every(A=>A.passed)&&X&&h.jsx(Nt,{to:`/problem/${e}/${X}`,className:"px-4 py-1.5 bg-green-100 text-green-700 text-sm font-medium rounded-md hover:bg-green-200 transition-colors",children:"Next Problem →"}),h.jsx("button",{onClick:()=>x(!g),className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",title:g?"Restore layout":"Maximize editor",children:g?"⊟ Restore":"⊞ Maximize"}),h.jsx("button",{onClick:F,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Reset"})]})]}),h.jsxs(Pr,{className:"flex-1 overflow-hidden",direction:"vertical",sizes:[60,40],minSize:100,gutterSize:8,children:[h.jsx("div",{className:"h-full overflow-hidden p-4",children:h.jsx(pu,{value:p,onChange:N,height:"100%",onRun:I})}),h.jsxs("div",{className:"overflow-auto p-4 space-y-4 bg-gray-50",children:[h.jsx(of,{results:y,isRunning:k}),h.jsx(fu,{output:s,isLoading:k,testCases:C,onTestCasesChange:z,problemId:t,resetKey:S})]})]})]})]})]})}const sf="ml-interview-scratchpad",lf=`import numpy as np

# Try out your code here!
# NumPy is already loaded and ready to use.

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {np.sum(arr)}")
print(f"Mean: {np.mean(arr)}")
`;function dz(){const{isLoading:e,isReady:t,output:n,runCode:r,clearOutput:i}=Ng(),[a,o]=E.useState(""),[s,l]=E.useState(!1);E.useEffect(()=>{const p=localStorage.getItem(sf);o(p||lf)},[]),E.useEffect(()=>{if(a){const p=setTimeout(()=>{localStorage.setItem(sf,a)},1e3);return()=>clearTimeout(p)}},[a]);const u=E.useCallback(p=>{o(p)},[]),d=E.useCallback(()=>{o(lf),i()},[i]),c=E.useCallback(async()=>{if(t){l(!0),i();try{await r(a)}finally{l(!1)}}},[t,a,r,i]);return h.jsxs("div",{className:"h-[calc(100vh-8rem)] flex flex-col -m-6",children:[h.jsx(Qo,{title:"Python Scratchpad",description:"Free Python playground with NumPy pre-loaded. Run Python code directly in your browser - no setup required. Practice machine learning code instantly.",canonical:"/scratchpad",keywords:"Python playground, online Python, NumPy, code sandbox, ML practice, browser Python"}),h.jsxs("div",{className:"flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200",children:[h.jsxs("div",{className:"flex items-center gap-3",children:[h.jsx(Nt,{to:"/",className:"text-gray-500 hover:text-gray-900 transition-colors",children:"Home"}),h.jsx("span",{className:"text-gray-300",children:"/"}),h.jsx("span",{className:"text-gray-900 font-medium",children:"Python Scratchpad"})]}),h.jsxs("div",{className:"flex items-center gap-3",children:[!t&&h.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[h.jsx("div",{className:"w-2 h-2 bg-yellow-500 rounded-full animate-pulse"}),e?"Loading Python...":"Python ready"]}),t&&h.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[h.jsx("div",{className:"w-2 h-2 bg-green-500 rounded-full"}),"Python ready"]})]})]}),h.jsxs("div",{className:"flex flex-col flex-1 overflow-hidden bg-white",children:[h.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200",children:[h.jsxs("div",{className:"flex items-center gap-2",children:[h.jsx("button",{onClick:c,disabled:!t||s,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:s?"Running...":"Run Code"}),h.jsx("button",{onClick:d,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Reset"}),h.jsx("button",{onClick:i,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Clear Output"})]}),h.jsx("span",{className:"text-gray-400 text-sm",children:"Code is auto-saved"})]}),h.jsxs(Pr,{className:"flex-1 overflow-hidden",direction:"vertical",sizes:[60,40],minSize:100,gutterSize:8,children:[h.jsx("div",{className:"h-full overflow-hidden p-4",children:h.jsx(pu,{value:a,onChange:u,height:"100%"})}),h.jsx("div",{className:"overflow-auto p-4 bg-gray-50",children:h.jsx(fu,{output:n,isLoading:s})})]})]})]})}function pz(){return h.jsx(Pv,{children:h.jsxs(R1,{children:[h.jsx(di,{path:"/",element:h.jsx(jv,{})}),h.jsx(di,{path:"/section/:sectionId",element:h.jsx(A2,{})}),h.jsx(di,{path:"/problem/:sectionId/:problemId",element:h.jsx(cz,{})}),h.jsx(di,{path:"/scratchpad",element:h.jsx(dz,{})})]})})}el.createRoot(document.getElementById("root")).render(h.jsx(Ne.StrictMode,{children:h.jsx(qh,{children:h.jsx(B1,{basename:"/ml-coding-lab",children:h.jsx(Sv,{children:h.jsx(pz,{})})})})}));
