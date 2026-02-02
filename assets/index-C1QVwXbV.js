var my=Object.defineProperty;var hy=(e,t,n)=>t in e?my(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n;var $t=(e,t,n)=>hy(e,typeof t!="symbol"?t+"":t,n);function gy(e,t){for(var n=0;n<t.length;n++){const r=t[n];if(typeof r!="string"&&!Array.isArray(r)){for(const i in r)if(i!=="default"&&!(i in e)){const a=Object.getOwnPropertyDescriptor(r,i);a&&Object.defineProperty(e,i,a.get?a:{enumerable:!0,get:()=>r[i]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const a of i)if(a.type==="childList")for(const o of a.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&r(o)}).observe(document,{childList:!0,subtree:!0});function n(i){const a={};return i.integrity&&(a.integrity=i.integrity),i.referrerPolicy&&(a.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?a.credentials="include":i.crossOrigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function r(i){if(i.ep)return;i.ep=!0;const a=n(i);fetch(i.href,a)}})();var Xa=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function Jn(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}var lf={exports:{}},Po={},uf={exports:{}},J={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Xi=Symbol.for("react.element"),yy=Symbol.for("react.portal"),xy=Symbol.for("react.fragment"),_y=Symbol.for("react.strict_mode"),vy=Symbol.for("react.profiler"),by=Symbol.for("react.provider"),wy=Symbol.for("react.context"),ky=Symbol.for("react.forward_ref"),Sy=Symbol.for("react.suspense"),Cy=Symbol.for("react.memo"),Ey=Symbol.for("react.lazy"),Lc=Symbol.iterator;function zy(e){return e===null||typeof e!="object"?null:(e=Lc&&e[Lc]||e["@@iterator"],typeof e=="function"?e:null)}var cf={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},df=Object.assign,pf={};function Wr(e,t,n){this.props=e,this.context=t,this.refs=pf,this.updater=n||cf}Wr.prototype.isReactComponent={};Wr.prototype.setState=function(e,t){if(typeof e!="object"&&typeof e!="function"&&e!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,e,t,"setState")};Wr.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,"forceUpdate")};function ff(){}ff.prototype=Wr.prototype;function mu(e,t,n){this.props=e,this.context=t,this.refs=pf,this.updater=n||cf}var hu=mu.prototype=new ff;hu.constructor=mu;df(hu,Wr.prototype);hu.isPureReactComponent=!0;var Ac=Array.isArray,mf=Object.prototype.hasOwnProperty,gu={current:null},hf={key:!0,ref:!0,__self:!0,__source:!0};function gf(e,t,n){var r,i={},a=null,o=null;if(t!=null)for(r in t.ref!==void 0&&(o=t.ref),t.key!==void 0&&(a=""+t.key),t)mf.call(t,r)&&!hf.hasOwnProperty(r)&&(i[r]=t[r]);var s=arguments.length-2;if(s===1)i.children=n;else if(1<s){for(var l=Array(s),u=0;u<s;u++)l[u]=arguments[u+2];i.children=l}if(e&&e.defaultProps)for(r in s=e.defaultProps,s)i[r]===void 0&&(i[r]=s[r]);return{$$typeof:Xi,type:e,key:a,ref:o,props:i,_owner:gu.current}}function Ty(e,t){return{$$typeof:Xi,type:e.type,key:t,ref:e.ref,props:e.props,_owner:e._owner}}function yu(e){return typeof e=="object"&&e!==null&&e.$$typeof===Xi}function Py(e){var t={"=":"=0",":":"=2"};return"$"+e.replace(/[=:]/g,function(n){return t[n]})}var Rc=/\/+/g;function ts(e,t){return typeof e=="object"&&e!==null&&e.key!=null?Py(""+e.key):t.toString(36)}function Ra(e,t,n,r,i){var a=typeof e;(a==="undefined"||a==="boolean")&&(e=null);var o=!1;if(e===null)o=!0;else switch(a){case"string":case"number":o=!0;break;case"object":switch(e.$$typeof){case Xi:case yy:o=!0}}if(o)return o=e,i=i(o),e=r===""?"."+ts(o,0):r,Ac(i)?(n="",e!=null&&(n=e.replace(Rc,"$&/")+"/"),Ra(i,t,n,"",function(u){return u})):i!=null&&(yu(i)&&(i=Ty(i,n+(!i.key||o&&o.key===i.key?"":(""+i.key).replace(Rc,"$&/")+"/")+e)),t.push(i)),1;if(o=0,r=r===""?".":r+":",Ac(e))for(var s=0;s<e.length;s++){a=e[s];var l=r+ts(a,s);o+=Ra(a,t,n,l,i)}else if(l=zy(e),typeof l=="function")for(e=l.call(e),s=0;!(a=e.next()).done;)a=a.value,l=r+ts(a,s++),o+=Ra(a,t,n,l,i);else if(a==="object")throw t=String(e),Error("Objects are not valid as a React child (found: "+(t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)+"). If you meant to render a collection of children, use an array instead.");return o}function la(e,t,n){if(e==null)return e;var r=[],i=0;return Ra(e,r,"","",function(a){return t.call(n,a,i++)}),r}function Ny(e){if(e._status===-1){var t=e._result;t=t(),t.then(function(n){(e._status===0||e._status===-1)&&(e._status=1,e._result=n)},function(n){(e._status===0||e._status===-1)&&(e._status=2,e._result=n)}),e._status===-1&&(e._status=0,e._result=t)}if(e._status===1)return e._result.default;throw e._result}var $e={current:null},Ia={transition:null},jy={ReactCurrentDispatcher:$e,ReactCurrentBatchConfig:Ia,ReactCurrentOwner:gu};function yf(){throw Error("act(...) is not supported in production builds of React.")}J.Children={map:la,forEach:function(e,t,n){la(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return la(e,function(){t++}),t},toArray:function(e){return la(e,function(t){return t})||[]},only:function(e){if(!yu(e))throw Error("React.Children.only expected to receive a single React element child.");return e}};J.Component=Wr;J.Fragment=xy;J.Profiler=vy;J.PureComponent=mu;J.StrictMode=_y;J.Suspense=Sy;J.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=jy;J.act=yf;J.cloneElement=function(e,t,n){if(e==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+e+".");var r=df({},e.props),i=e.key,a=e.ref,o=e._owner;if(t!=null){if(t.ref!==void 0&&(a=t.ref,o=gu.current),t.key!==void 0&&(i=""+t.key),e.type&&e.type.defaultProps)var s=e.type.defaultProps;for(l in t)mf.call(t,l)&&!hf.hasOwnProperty(l)&&(r[l]=t[l]===void 0&&s!==void 0?s[l]:t[l])}var l=arguments.length-2;if(l===1)r.children=n;else if(1<l){s=Array(l);for(var u=0;u<l;u++)s[u]=arguments[u+2];r.children=s}return{$$typeof:Xi,type:e.type,key:i,ref:a,props:r,_owner:o}};J.createContext=function(e){return e={$$typeof:wy,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},e.Provider={$$typeof:by,_context:e},e.Consumer=e};J.createElement=gf;J.createFactory=function(e){var t=gf.bind(null,e);return t.type=e,t};J.createRef=function(){return{current:null}};J.forwardRef=function(e){return{$$typeof:ky,render:e}};J.isValidElement=yu;J.lazy=function(e){return{$$typeof:Ey,_payload:{_status:-1,_result:e},_init:Ny}};J.memo=function(e,t){return{$$typeof:Cy,type:e,compare:t===void 0?null:t}};J.startTransition=function(e){var t=Ia.transition;Ia.transition={};try{e()}finally{Ia.transition=t}};J.unstable_act=yf;J.useCallback=function(e,t){return $e.current.useCallback(e,t)};J.useContext=function(e){return $e.current.useContext(e)};J.useDebugValue=function(){};J.useDeferredValue=function(e){return $e.current.useDeferredValue(e)};J.useEffect=function(e,t){return $e.current.useEffect(e,t)};J.useId=function(){return $e.current.useId()};J.useImperativeHandle=function(e,t,n){return $e.current.useImperativeHandle(e,t,n)};J.useInsertionEffect=function(e,t){return $e.current.useInsertionEffect(e,t)};J.useLayoutEffect=function(e,t){return $e.current.useLayoutEffect(e,t)};J.useMemo=function(e,t){return $e.current.useMemo(e,t)};J.useReducer=function(e,t,n){return $e.current.useReducer(e,t,n)};J.useRef=function(e){return $e.current.useRef(e)};J.useState=function(e){return $e.current.useState(e)};J.useSyncExternalStore=function(e,t,n){return $e.current.useSyncExternalStore(e,t,n)};J.useTransition=function(){return $e.current.useTransition()};J.version="18.3.1";uf.exports=J;var E=uf.exports;const Ne=Jn(E),Ly=gy({__proto__:null,default:Ne},[E]);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Ay=E,Ry=Symbol.for("react.element"),Iy=Symbol.for("react.fragment"),Oy=Object.prototype.hasOwnProperty,Dy=Ay.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,Fy={key:!0,ref:!0,__self:!0,__source:!0};function xf(e,t,n){var r,i={},a=null,o=null;n!==void 0&&(a=""+n),t.key!==void 0&&(a=""+t.key),t.ref!==void 0&&(o=t.ref);for(r in t)Oy.call(t,r)&&!Fy.hasOwnProperty(r)&&(i[r]=t[r]);if(e&&e.defaultProps)for(r in t=e.defaultProps,t)i[r]===void 0&&(i[r]=t[r]);return{$$typeof:Ry,type:e,key:a,ref:o,props:i,_owner:Dy.current}}Po.Fragment=Iy;Po.jsx=xf;Po.jsxs=xf;lf.exports=Po;var p=lf.exports,Js={},_f={exports:{}},ct={},vf={exports:{}},bf={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(e){function t(O,U){var v=O.length;O.push(U);e:for(;0<v;){var B=v-1>>>1,Q=O[B];if(0<i(Q,U))O[B]=U,O[v]=Q,v=B;else break e}}function n(O){return O.length===0?null:O[0]}function r(O){if(O.length===0)return null;var U=O[0],v=O.pop();if(v!==U){O[0]=v;e:for(var B=0,Q=O.length,b=Q>>>1;B<b;){var ie=2*(B+1)-1,we=O[ie],le=ie+1,pt=O[le];if(0>i(we,v))le<Q&&0>i(pt,we)?(O[B]=pt,O[le]=v,B=le):(O[B]=we,O[ie]=v,B=ie);else if(le<Q&&0>i(pt,v))O[B]=pt,O[le]=v,B=le;else break e}}return U}function i(O,U){var v=O.sortIndex-U.sortIndex;return v!==0?v:O.id-U.id}if(typeof performance=="object"&&typeof performance.now=="function"){var a=performance;e.unstable_now=function(){return a.now()}}else{var o=Date,s=o.now();e.unstable_now=function(){return o.now()-s}}var l=[],u=[],d=1,c=null,f=3,m=!1,y=!1,_=!1,k=typeof setTimeout=="function"?setTimeout:null,h=typeof clearTimeout=="function"?clearTimeout:null,g=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function x(O){for(var U=n(u);U!==null;){if(U.callback===null)r(u);else if(U.startTime<=O)r(u),U.sortIndex=U.expirationTime,t(l,U);else break;U=n(u)}}function C(O){if(_=!1,x(O),!y)if(n(l)!==null)y=!0,$(z);else{var U=n(u);U!==null&&ne(C,U.startTime-O)}}function z(O,U){y=!1,_&&(_=!1,h(N),N=-1),m=!0;var v=f;try{for(x(U),c=n(l);c!==null&&(!(c.expirationTime>U)||O&&!I());){var B=c.callback;if(typeof B=="function"){c.callback=null,f=c.priorityLevel;var Q=B(c.expirationTime<=U);U=e.unstable_now(),typeof Q=="function"?c.callback=Q:c===n(l)&&r(l),x(U)}else r(l);c=n(l)}if(c!==null)var b=!0;else{var ie=n(u);ie!==null&&ne(C,ie.startTime-U),b=!1}return b}finally{c=null,f=v,m=!1}}var S=!1,T=null,N=-1,M=5,w=-1;function I(){return!(e.unstable_now()-w<M)}function W(){if(T!==null){var O=e.unstable_now();w=O;var U=!0;try{U=T(!0,O)}finally{U?Y():(S=!1,T=null)}}else S=!1}var Y;if(typeof g=="function")Y=function(){g(W)};else if(typeof MessageChannel<"u"){var ee=new MessageChannel,j=ee.port2;ee.port1.onmessage=W,Y=function(){j.postMessage(null)}}else Y=function(){k(W,0)};function $(O){T=O,S||(S=!0,Y())}function ne(O,U){N=k(function(){O(e.unstable_now())},U)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(O){O.callback=null},e.unstable_continueExecution=function(){y||m||(y=!0,$(z))},e.unstable_forceFrameRate=function(O){0>O||125<O?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):M=0<O?Math.floor(1e3/O):5},e.unstable_getCurrentPriorityLevel=function(){return f},e.unstable_getFirstCallbackNode=function(){return n(l)},e.unstable_next=function(O){switch(f){case 1:case 2:case 3:var U=3;break;default:U=f}var v=f;f=U;try{return O()}finally{f=v}},e.unstable_pauseExecution=function(){},e.unstable_requestPaint=function(){},e.unstable_runWithPriority=function(O,U){switch(O){case 1:case 2:case 3:case 4:case 5:break;default:O=3}var v=f;f=O;try{return U()}finally{f=v}},e.unstable_scheduleCallback=function(O,U,v){var B=e.unstable_now();switch(typeof v=="object"&&v!==null?(v=v.delay,v=typeof v=="number"&&0<v?B+v:B):v=B,O){case 1:var Q=-1;break;case 2:Q=250;break;case 5:Q=1073741823;break;case 4:Q=1e4;break;default:Q=5e3}return Q=v+Q,O={id:d++,callback:U,priorityLevel:O,startTime:v,expirationTime:Q,sortIndex:-1},v>B?(O.sortIndex=v,t(u,O),n(l)===null&&O===n(u)&&(_?(h(N),N=-1):_=!0,ne(C,v-B))):(O.sortIndex=Q,t(l,O),y||m||(y=!0,$(z))),O},e.unstable_shouldYield=I,e.unstable_wrapCallback=function(O){var U=f;return function(){var v=f;f=U;try{return O.apply(this,arguments)}finally{f=v}}}})(bf);vf.exports=bf;var My=vf.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var qy=E,ut=My;function A(e){for(var t="https://reactjs.org/docs/error-decoder.html?invariant="+e,n=1;n<arguments.length;n++)t+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+e+"; visit "+t+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var wf=new Set,zi={};function er(e,t){jr(e,t),jr(e+"Capture",t)}function jr(e,t){for(zi[e]=t,e=0;e<t.length;e++)wf.add(t[e])}var Jt=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),el=Object.prototype.hasOwnProperty,Wy=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,Ic={},Oc={};function By(e){return el.call(Oc,e)?!0:el.call(Ic,e)?!1:Wy.test(e)?Oc[e]=!0:(Ic[e]=!0,!1)}function Vy(e,t,n,r){if(n!==null&&n.type===0)return!1;switch(typeof t){case"function":case"symbol":return!0;case"boolean":return r?!1:n!==null?!n.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function Uy(e,t,n,r){if(t===null||typeof t>"u"||Vy(e,t,n,r))return!0;if(r)return!1;if(n!==null)switch(n.type){case 3:return!t;case 4:return t===!1;case 5:return isNaN(t);case 6:return isNaN(t)||1>t}return!1}function Qe(e,t,n,r,i,a,o){this.acceptsBooleans=t===2||t===3||t===4,this.attributeName=r,this.attributeNamespace=i,this.mustUseProperty=n,this.propertyName=e,this.type=t,this.sanitizeURL=a,this.removeEmptyString=o}var Re={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){Re[e]=new Qe(e,0,!1,e,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var t=e[0];Re[t]=new Qe(t,1,!1,e[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(e){Re[e]=new Qe(e,2,!1,e.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){Re[e]=new Qe(e,2,!1,e,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){Re[e]=new Qe(e,3,!1,e.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(e){Re[e]=new Qe(e,3,!0,e,null,!1,!1)});["capture","download"].forEach(function(e){Re[e]=new Qe(e,4,!1,e,null,!1,!1)});["cols","rows","size","span"].forEach(function(e){Re[e]=new Qe(e,6,!1,e,null,!1,!1)});["rowSpan","start"].forEach(function(e){Re[e]=new Qe(e,5,!1,e.toLowerCase(),null,!1,!1)});var xu=/[\-:]([a-z])/g;function _u(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var t=e.replace(xu,_u);Re[t]=new Qe(t,1,!1,e,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var t=e.replace(xu,_u);Re[t]=new Qe(t,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(e){var t=e.replace(xu,_u);Re[t]=new Qe(t,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(e){Re[e]=new Qe(e,1,!1,e.toLowerCase(),null,!1,!1)});Re.xlinkHref=new Qe("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(e){Re[e]=new Qe(e,1,!1,e.toLowerCase(),null,!0,!0)});function vu(e,t,n,r){var i=Re.hasOwnProperty(t)?Re[t]:null;(i!==null?i.type!==0:r||!(2<t.length)||t[0]!=="o"&&t[0]!=="O"||t[1]!=="n"&&t[1]!=="N")&&(Uy(t,n,i,r)&&(n=null),r||i===null?By(t)&&(n===null?e.removeAttribute(t):e.setAttribute(t,""+n)):i.mustUseProperty?e[i.propertyName]=n===null?i.type===3?!1:"":n:(t=i.attributeName,r=i.attributeNamespace,n===null?e.removeAttribute(t):(i=i.type,n=i===3||i===4&&n===!0?"":""+n,r?e.setAttributeNS(r,t,n):e.setAttribute(t,n))))}var rn=qy.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,ua=Symbol.for("react.element"),lr=Symbol.for("react.portal"),ur=Symbol.for("react.fragment"),bu=Symbol.for("react.strict_mode"),tl=Symbol.for("react.profiler"),kf=Symbol.for("react.provider"),Sf=Symbol.for("react.context"),wu=Symbol.for("react.forward_ref"),nl=Symbol.for("react.suspense"),rl=Symbol.for("react.suspense_list"),ku=Symbol.for("react.memo"),ln=Symbol.for("react.lazy"),Cf=Symbol.for("react.offscreen"),Dc=Symbol.iterator;function Xr(e){return e===null||typeof e!="object"?null:(e=Dc&&e[Dc]||e["@@iterator"],typeof e=="function"?e:null)}var xe=Object.assign,ns;function ui(e){if(ns===void 0)try{throw Error()}catch(n){var t=n.stack.trim().match(/\n( *(at )?)/);ns=t&&t[1]||""}return`
`+ns+e}var rs=!1;function is(e,t){if(!e||rs)return"";rs=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(t)if(t=function(){throw Error()},Object.defineProperty(t.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(t,[])}catch(u){var r=u}Reflect.construct(e,[],t)}else{try{t.call()}catch(u){r=u}e.call(t.prototype)}else{try{throw Error()}catch(u){r=u}e()}}catch(u){if(u&&r&&typeof u.stack=="string"){for(var i=u.stack.split(`
`),a=r.stack.split(`
`),o=i.length-1,s=a.length-1;1<=o&&0<=s&&i[o]!==a[s];)s--;for(;1<=o&&0<=s;o--,s--)if(i[o]!==a[s]){if(o!==1||s!==1)do if(o--,s--,0>s||i[o]!==a[s]){var l=`
`+i[o].replace(" at new "," at ");return e.displayName&&l.includes("<anonymous>")&&(l=l.replace("<anonymous>",e.displayName)),l}while(1<=o&&0<=s);break}}}finally{rs=!1,Error.prepareStackTrace=n}return(e=e?e.displayName||e.name:"")?ui(e):""}function Hy(e){switch(e.tag){case 5:return ui(e.type);case 16:return ui("Lazy");case 13:return ui("Suspense");case 19:return ui("SuspenseList");case 0:case 2:case 15:return e=is(e.type,!1),e;case 11:return e=is(e.type.render,!1),e;case 1:return e=is(e.type,!0),e;default:return""}}function il(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case ur:return"Fragment";case lr:return"Portal";case tl:return"Profiler";case bu:return"StrictMode";case nl:return"Suspense";case rl:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case Sf:return(e.displayName||"Context")+".Consumer";case kf:return(e._context.displayName||"Context")+".Provider";case wu:var t=e.render;return e=e.displayName,e||(e=t.displayName||t.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case ku:return t=e.displayName||null,t!==null?t:il(e.type)||"Memo";case ln:t=e._payload,e=e._init;try{return il(e(t))}catch{}}return null}function $y(e){var t=e.type;switch(e.tag){case 24:return"Cache";case 9:return(t.displayName||"Context")+".Consumer";case 10:return(t._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=t.render,e=e.displayName||e.name||"",t.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return t;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return il(t);case 8:return t===bu?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t}return null}function Cn(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function Ef(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(t==="checkbox"||t==="radio")}function Qy(e){var t=Ef(e)?"checked":"value",n=Object.getOwnPropertyDescriptor(e.constructor.prototype,t),r=""+e[t];if(!e.hasOwnProperty(t)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var i=n.get,a=n.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(o){r=""+o,a.call(this,o)}}),Object.defineProperty(e,t,{enumerable:n.enumerable}),{getValue:function(){return r},setValue:function(o){r=""+o},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function ca(e){e._valueTracker||(e._valueTracker=Qy(e))}function zf(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r="";return e&&(r=Ef(e)?e.checked?"true":"false":e.value),e=r,e!==n?(t.setValue(e),!0):!1}function Ga(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function al(e,t){var n=t.checked;return xe({},t,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??e._wrapperState.initialChecked})}function Fc(e,t){var n=t.defaultValue==null?"":t.defaultValue,r=t.checked!=null?t.checked:t.defaultChecked;n=Cn(t.value!=null?t.value:n),e._wrapperState={initialChecked:r,initialValue:n,controlled:t.type==="checkbox"||t.type==="radio"?t.checked!=null:t.value!=null}}function Tf(e,t){t=t.checked,t!=null&&vu(e,"checked",t,!1)}function ol(e,t){Tf(e,t);var n=Cn(t.value),r=t.type;if(n!=null)r==="number"?(n===0&&e.value===""||e.value!=n)&&(e.value=""+n):e.value!==""+n&&(e.value=""+n);else if(r==="submit"||r==="reset"){e.removeAttribute("value");return}t.hasOwnProperty("value")?sl(e,t.type,n):t.hasOwnProperty("defaultValue")&&sl(e,t.type,Cn(t.defaultValue)),t.checked==null&&t.defaultChecked!=null&&(e.defaultChecked=!!t.defaultChecked)}function Mc(e,t,n){if(t.hasOwnProperty("value")||t.hasOwnProperty("defaultValue")){var r=t.type;if(!(r!=="submit"&&r!=="reset"||t.value!==void 0&&t.value!==null))return;t=""+e._wrapperState.initialValue,n||t===e.value||(e.value=t),e.defaultValue=t}n=e.name,n!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,n!==""&&(e.name=n)}function sl(e,t,n){(t!=="number"||Ga(e.ownerDocument)!==e)&&(n==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+n&&(e.defaultValue=""+n))}var ci=Array.isArray;function br(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t["$"+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty("$"+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=""+Cn(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function ll(e,t){if(t.dangerouslySetInnerHTML!=null)throw Error(A(91));return xe({},t,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function qc(e,t){var n=t.value;if(n==null){if(n=t.children,t=t.defaultValue,n!=null){if(t!=null)throw Error(A(92));if(ci(n)){if(1<n.length)throw Error(A(93));n=n[0]}t=n}t==null&&(t=""),n=t}e._wrapperState={initialValue:Cn(n)}}function Pf(e,t){var n=Cn(t.value),r=Cn(t.defaultValue);n!=null&&(n=""+n,n!==e.value&&(e.value=n),t.defaultValue==null&&e.defaultValue!==n&&(e.defaultValue=n)),r!=null&&(e.defaultValue=""+r)}function Wc(e){var t=e.textContent;t===e._wrapperState.initialValue&&t!==""&&t!==null&&(e.value=t)}function Nf(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function ul(e,t){return e==null||e==="http://www.w3.org/1999/xhtml"?Nf(t):e==="http://www.w3.org/2000/svg"&&t==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var da,jf=function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(t,n,r,i){MSApp.execUnsafeLocalFunction(function(){return e(t,n,r,i)})}:e}(function(e,t){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=t;else{for(da=da||document.createElement("div"),da.innerHTML="<svg>"+t.valueOf().toString()+"</svg>",t=da.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;t.firstChild;)e.appendChild(t.firstChild)}});function Ti(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var mi={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},Ky=["Webkit","ms","Moz","O"];Object.keys(mi).forEach(function(e){Ky.forEach(function(t){t=t+e.charAt(0).toUpperCase()+e.substring(1),mi[t]=mi[e]})});function Lf(e,t,n){return t==null||typeof t=="boolean"||t===""?"":n||typeof t!="number"||t===0||mi.hasOwnProperty(e)&&mi[e]?(""+t).trim():t+"px"}function Af(e,t){e=e.style;for(var n in t)if(t.hasOwnProperty(n)){var r=n.indexOf("--")===0,i=Lf(n,t[n],r);n==="float"&&(n="cssFloat"),r?e.setProperty(n,i):e[n]=i}}var Yy=xe({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function cl(e,t){if(t){if(Yy[e]&&(t.children!=null||t.dangerouslySetInnerHTML!=null))throw Error(A(137,e));if(t.dangerouslySetInnerHTML!=null){if(t.children!=null)throw Error(A(60));if(typeof t.dangerouslySetInnerHTML!="object"||!("__html"in t.dangerouslySetInnerHTML))throw Error(A(61))}if(t.style!=null&&typeof t.style!="object")throw Error(A(62))}}function dl(e,t){if(e.indexOf("-")===-1)return typeof t.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var pl=null;function Su(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var fl=null,wr=null,kr=null;function Bc(e){if(e=Ji(e)){if(typeof fl!="function")throw Error(A(280));var t=e.stateNode;t&&(t=Ro(t),fl(e.stateNode,e.type,t))}}function Rf(e){wr?kr?kr.push(e):kr=[e]:wr=e}function If(){if(wr){var e=wr,t=kr;if(kr=wr=null,Bc(e),t)for(e=0;e<t.length;e++)Bc(t[e])}}function Of(e,t){return e(t)}function Df(){}var as=!1;function Ff(e,t,n){if(as)return e(t,n);as=!0;try{return Of(e,t,n)}finally{as=!1,(wr!==null||kr!==null)&&(Df(),If())}}function Pi(e,t){var n=e.stateNode;if(n===null)return null;var r=Ro(n);if(r===null)return null;n=r[t];e:switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(r=!r.disabled)||(e=e.type,r=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!r;break e;default:e=!1}if(e)return null;if(n&&typeof n!="function")throw Error(A(231,t,typeof n));return n}var ml=!1;if(Jt)try{var Gr={};Object.defineProperty(Gr,"passive",{get:function(){ml=!0}}),window.addEventListener("test",Gr,Gr),window.removeEventListener("test",Gr,Gr)}catch{ml=!1}function Xy(e,t,n,r,i,a,o,s,l){var u=Array.prototype.slice.call(arguments,3);try{t.apply(n,u)}catch(d){this.onError(d)}}var hi=!1,Za=null,Ja=!1,hl=null,Gy={onError:function(e){hi=!0,Za=e}};function Zy(e,t,n,r,i,a,o,s,l){hi=!1,Za=null,Xy.apply(Gy,arguments)}function Jy(e,t,n,r,i,a,o,s,l){if(Zy.apply(this,arguments),hi){if(hi){var u=Za;hi=!1,Za=null}else throw Error(A(198));Ja||(Ja=!0,hl=u)}}function tr(e){var t=e,n=e;if(e.alternate)for(;t.return;)t=t.return;else{e=t;do t=e,t.flags&4098&&(n=t.return),e=t.return;while(e)}return t.tag===3?n:null}function Mf(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function Vc(e){if(tr(e)!==e)throw Error(A(188))}function e0(e){var t=e.alternate;if(!t){if(t=tr(e),t===null)throw Error(A(188));return t!==e?null:e}for(var n=e,r=t;;){var i=n.return;if(i===null)break;var a=i.alternate;if(a===null){if(r=i.return,r!==null){n=r;continue}break}if(i.child===a.child){for(a=i.child;a;){if(a===n)return Vc(i),e;if(a===r)return Vc(i),t;a=a.sibling}throw Error(A(188))}if(n.return!==r.return)n=i,r=a;else{for(var o=!1,s=i.child;s;){if(s===n){o=!0,n=i,r=a;break}if(s===r){o=!0,r=i,n=a;break}s=s.sibling}if(!o){for(s=a.child;s;){if(s===n){o=!0,n=a,r=i;break}if(s===r){o=!0,r=a,n=i;break}s=s.sibling}if(!o)throw Error(A(189))}}if(n.alternate!==r)throw Error(A(190))}if(n.tag!==3)throw Error(A(188));return n.stateNode.current===n?e:t}function qf(e){return e=e0(e),e!==null?Wf(e):null}function Wf(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var t=Wf(e);if(t!==null)return t;e=e.sibling}return null}var Bf=ut.unstable_scheduleCallback,Uc=ut.unstable_cancelCallback,t0=ut.unstable_shouldYield,n0=ut.unstable_requestPaint,ke=ut.unstable_now,r0=ut.unstable_getCurrentPriorityLevel,Cu=ut.unstable_ImmediatePriority,Vf=ut.unstable_UserBlockingPriority,eo=ut.unstable_NormalPriority,i0=ut.unstable_LowPriority,Uf=ut.unstable_IdlePriority,No=null,Wt=null;function a0(e){if(Wt&&typeof Wt.onCommitFiberRoot=="function")try{Wt.onCommitFiberRoot(No,e,void 0,(e.current.flags&128)===128)}catch{}}var Lt=Math.clz32?Math.clz32:l0,o0=Math.log,s0=Math.LN2;function l0(e){return e>>>=0,e===0?32:31-(o0(e)/s0|0)|0}var pa=64,fa=4194304;function di(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function to(e,t){var n=e.pendingLanes;if(n===0)return 0;var r=0,i=e.suspendedLanes,a=e.pingedLanes,o=n&268435455;if(o!==0){var s=o&~i;s!==0?r=di(s):(a&=o,a!==0&&(r=di(a)))}else o=n&~i,o!==0?r=di(o):a!==0&&(r=di(a));if(r===0)return 0;if(t!==0&&t!==r&&!(t&i)&&(i=r&-r,a=t&-t,i>=a||i===16&&(a&4194240)!==0))return t;if(r&4&&(r|=n&16),t=e.entangledLanes,t!==0)for(e=e.entanglements,t&=r;0<t;)n=31-Lt(t),i=1<<n,r|=e[n],t&=~i;return r}function u0(e,t){switch(e){case 1:case 2:case 4:return t+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function c0(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,a=e.pendingLanes;0<a;){var o=31-Lt(a),s=1<<o,l=i[o];l===-1?(!(s&n)||s&r)&&(i[o]=u0(s,t)):l<=t&&(e.expiredLanes|=s),a&=~s}}function gl(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function Hf(){var e=pa;return pa<<=1,!(pa&4194240)&&(pa=64),e}function os(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function Gi(e,t,n){e.pendingLanes|=t,t!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,t=31-Lt(t),e[t]=n}function d0(e,t){var n=e.pendingLanes&~t;e.pendingLanes=t,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=t,e.mutableReadLanes&=t,e.entangledLanes&=t,t=e.entanglements;var r=e.eventTimes;for(e=e.expirationTimes;0<n;){var i=31-Lt(n),a=1<<i;t[i]=0,r[i]=-1,e[i]=-1,n&=~a}}function Eu(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-Lt(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}var se=0;function $f(e){return e&=-e,1<e?4<e?e&268435455?16:536870912:4:1}var Qf,zu,Kf,Yf,Xf,yl=!1,ma=[],hn=null,gn=null,yn=null,Ni=new Map,ji=new Map,cn=[],p0="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function Hc(e,t){switch(e){case"focusin":case"focusout":hn=null;break;case"dragenter":case"dragleave":gn=null;break;case"mouseover":case"mouseout":yn=null;break;case"pointerover":case"pointerout":Ni.delete(t.pointerId);break;case"gotpointercapture":case"lostpointercapture":ji.delete(t.pointerId)}}function Zr(e,t,n,r,i,a){return e===null||e.nativeEvent!==a?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:a,targetContainers:[i]},t!==null&&(t=Ji(t),t!==null&&zu(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function f0(e,t,n,r,i){switch(t){case"focusin":return hn=Zr(hn,e,t,n,r,i),!0;case"dragenter":return gn=Zr(gn,e,t,n,r,i),!0;case"mouseover":return yn=Zr(yn,e,t,n,r,i),!0;case"pointerover":var a=i.pointerId;return Ni.set(a,Zr(Ni.get(a)||null,e,t,n,r,i)),!0;case"gotpointercapture":return a=i.pointerId,ji.set(a,Zr(ji.get(a)||null,e,t,n,r,i)),!0}return!1}function Gf(e){var t=Wn(e.target);if(t!==null){var n=tr(t);if(n!==null){if(t=n.tag,t===13){if(t=Mf(n),t!==null){e.blockedOn=t,Xf(e.priority,function(){Kf(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Oa(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=xl(e.domEventName,e.eventSystemFlags,t[0],e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);pl=r,n.target.dispatchEvent(r),pl=null}else return t=Ji(n),t!==null&&zu(t),e.blockedOn=n,!1;t.shift()}return!0}function $c(e,t,n){Oa(e)&&n.delete(t)}function m0(){yl=!1,hn!==null&&Oa(hn)&&(hn=null),gn!==null&&Oa(gn)&&(gn=null),yn!==null&&Oa(yn)&&(yn=null),Ni.forEach($c),ji.forEach($c)}function Jr(e,t){e.blockedOn===t&&(e.blockedOn=null,yl||(yl=!0,ut.unstable_scheduleCallback(ut.unstable_NormalPriority,m0)))}function Li(e){function t(i){return Jr(i,e)}if(0<ma.length){Jr(ma[0],e);for(var n=1;n<ma.length;n++){var r=ma[n];r.blockedOn===e&&(r.blockedOn=null)}}for(hn!==null&&Jr(hn,e),gn!==null&&Jr(gn,e),yn!==null&&Jr(yn,e),Ni.forEach(t),ji.forEach(t),n=0;n<cn.length;n++)r=cn[n],r.blockedOn===e&&(r.blockedOn=null);for(;0<cn.length&&(n=cn[0],n.blockedOn===null);)Gf(n),n.blockedOn===null&&cn.shift()}var Sr=rn.ReactCurrentBatchConfig,no=!0;function h0(e,t,n,r){var i=se,a=Sr.transition;Sr.transition=null;try{se=1,Tu(e,t,n,r)}finally{se=i,Sr.transition=a}}function g0(e,t,n,r){var i=se,a=Sr.transition;Sr.transition=null;try{se=4,Tu(e,t,n,r)}finally{se=i,Sr.transition=a}}function Tu(e,t,n,r){if(no){var i=xl(e,t,n,r);if(i===null)gs(e,t,r,ro,n),Hc(e,r);else if(f0(i,e,t,n,r))r.stopPropagation();else if(Hc(e,r),t&4&&-1<p0.indexOf(e)){for(;i!==null;){var a=Ji(i);if(a!==null&&Qf(a),a=xl(e,t,n,r),a===null&&gs(e,t,r,ro,n),a===i)break;i=a}i!==null&&r.stopPropagation()}else gs(e,t,r,null,n)}}var ro=null;function xl(e,t,n,r){if(ro=null,e=Su(r),e=Wn(e),e!==null)if(t=tr(e),t===null)e=null;else if(n=t.tag,n===13){if(e=Mf(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null);return ro=e,null}function Zf(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(r0()){case Cu:return 1;case Vf:return 4;case eo:case i0:return 16;case Uf:return 536870912;default:return 16}default:return 16}}var pn=null,Pu=null,Da=null;function Jf(){if(Da)return Da;var e,t=Pu,n=t.length,r,i="value"in pn?pn.value:pn.textContent,a=i.length;for(e=0;e<n&&t[e]===i[e];e++);var o=n-e;for(r=1;r<=o&&t[n-r]===i[a-r];r++);return Da=i.slice(e,1<r?1-r:void 0)}function Fa(e){var t=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function ha(){return!0}function Qc(){return!1}function dt(e){function t(n,r,i,a,o){this._reactName=n,this._targetInst=i,this.type=r,this.nativeEvent=a,this.target=o,this.currentTarget=null;for(var s in e)e.hasOwnProperty(s)&&(n=e[s],this[s]=n?n(a):a[s]);return this.isDefaultPrevented=(a.defaultPrevented!=null?a.defaultPrevented:a.returnValue===!1)?ha:Qc,this.isPropagationStopped=Qc,this}return xe(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=ha)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=ha)},persist:function(){},isPersistent:ha}),t}var Br={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Nu=dt(Br),Zi=xe({},Br,{view:0,detail:0}),y0=dt(Zi),ss,ls,ei,jo=xe({},Zi,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:ju,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==ei&&(ei&&e.type==="mousemove"?(ss=e.screenX-ei.screenX,ls=e.screenY-ei.screenY):ls=ss=0,ei=e),ss)},movementY:function(e){return"movementY"in e?e.movementY:ls}}),Kc=dt(jo),x0=xe({},jo,{dataTransfer:0}),_0=dt(x0),v0=xe({},Zi,{relatedTarget:0}),us=dt(v0),b0=xe({},Br,{animationName:0,elapsedTime:0,pseudoElement:0}),w0=dt(b0),k0=xe({},Br,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),S0=dt(k0),C0=xe({},Br,{data:0}),Yc=dt(C0),E0={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},z0={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},T0={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function P0(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=T0[e])?!!t[e]:!1}function ju(){return P0}var N0=xe({},Zi,{key:function(e){if(e.key){var t=E0[e.key]||e.key;if(t!=="Unidentified")return t}return e.type==="keypress"?(e=Fa(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?z0[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:ju,charCode:function(e){return e.type==="keypress"?Fa(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?Fa(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),j0=dt(N0),L0=xe({},jo,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),Xc=dt(L0),A0=xe({},Zi,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:ju}),R0=dt(A0),I0=xe({},Br,{propertyName:0,elapsedTime:0,pseudoElement:0}),O0=dt(I0),D0=xe({},jo,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),F0=dt(D0),M0=[9,13,27,32],Lu=Jt&&"CompositionEvent"in window,gi=null;Jt&&"documentMode"in document&&(gi=document.documentMode);var q0=Jt&&"TextEvent"in window&&!gi,em=Jt&&(!Lu||gi&&8<gi&&11>=gi),Gc=" ",Zc=!1;function tm(e,t){switch(e){case"keyup":return M0.indexOf(t.keyCode)!==-1;case"keydown":return t.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function nm(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var cr=!1;function W0(e,t){switch(e){case"compositionend":return nm(t);case"keypress":return t.which!==32?null:(Zc=!0,Gc);case"textInput":return e=t.data,e===Gc&&Zc?null:e;default:return null}}function B0(e,t){if(cr)return e==="compositionend"||!Lu&&tm(e,t)?(e=Jf(),Da=Pu=pn=null,cr=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case"compositionend":return em&&t.locale!=="ko"?null:t.data;default:return null}}var V0={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Jc(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t==="input"?!!V0[e.type]:t==="textarea"}function rm(e,t,n,r){Rf(r),t=io(t,"onChange"),0<t.length&&(n=new Nu("onChange","change",null,n,r),e.push({event:n,listeners:t}))}var yi=null,Ai=null;function U0(e){mm(e,0)}function Lo(e){var t=fr(e);if(zf(t))return e}function H0(e,t){if(e==="change")return t}var im=!1;if(Jt){var cs;if(Jt){var ds="oninput"in document;if(!ds){var ed=document.createElement("div");ed.setAttribute("oninput","return;"),ds=typeof ed.oninput=="function"}cs=ds}else cs=!1;im=cs&&(!document.documentMode||9<document.documentMode)}function td(){yi&&(yi.detachEvent("onpropertychange",am),Ai=yi=null)}function am(e){if(e.propertyName==="value"&&Lo(Ai)){var t=[];rm(t,Ai,e,Su(e)),Ff(U0,t)}}function $0(e,t,n){e==="focusin"?(td(),yi=t,Ai=n,yi.attachEvent("onpropertychange",am)):e==="focusout"&&td()}function Q0(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return Lo(Ai)}function K0(e,t){if(e==="click")return Lo(t)}function Y0(e,t){if(e==="input"||e==="change")return Lo(t)}function X0(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var It=typeof Object.is=="function"?Object.is:X0;function Ri(e,t){if(It(e,t))return!0;if(typeof e!="object"||e===null||typeof t!="object"||t===null)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!el.call(t,i)||!It(e[i],t[i]))return!1}return!0}function nd(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function rd(e,t){var n=nd(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=nd(n)}}function om(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?om(e,t.parentNode):"contains"in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function sm(){for(var e=window,t=Ga();t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href=="string"}catch{n=!1}if(n)e=t.contentWindow;else break;t=Ga(e.document)}return t}function Au(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||t==="textarea"||e.contentEditable==="true")}function G0(e){var t=sm(),n=e.focusedElem,r=e.selectionRange;if(t!==n&&n&&n.ownerDocument&&om(n.ownerDocument.documentElement,n)){if(r!==null&&Au(n)){if(t=r.start,e=r.end,e===void 0&&(e=t),"selectionStart"in n)n.selectionStart=t,n.selectionEnd=Math.min(e,n.value.length);else if(e=(t=n.ownerDocument||document)&&t.defaultView||window,e.getSelection){e=e.getSelection();var i=n.textContent.length,a=Math.min(r.start,i);r=r.end===void 0?a:Math.min(r.end,i),!e.extend&&a>r&&(i=r,r=a,a=i),i=rd(n,a);var o=rd(n,r);i&&o&&(e.rangeCount!==1||e.anchorNode!==i.node||e.anchorOffset!==i.offset||e.focusNode!==o.node||e.focusOffset!==o.offset)&&(t=t.createRange(),t.setStart(i.node,i.offset),e.removeAllRanges(),a>r?(e.addRange(t),e.extend(o.node,o.offset)):(t.setEnd(o.node,o.offset),e.addRange(t)))}}for(t=[],e=n;e=e.parentNode;)e.nodeType===1&&t.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<t.length;n++)e=t[n],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var Z0=Jt&&"documentMode"in document&&11>=document.documentMode,dr=null,_l=null,xi=null,vl=!1;function id(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;vl||dr==null||dr!==Ga(r)||(r=dr,"selectionStart"in r&&Au(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),xi&&Ri(xi,r)||(xi=r,r=io(_l,"onSelect"),0<r.length&&(t=new Nu("onSelect","select",null,t,n),e.push({event:t,listeners:r}),t.target=dr)))}function ga(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n["Webkit"+e]="webkit"+t,n["Moz"+e]="moz"+t,n}var pr={animationend:ga("Animation","AnimationEnd"),animationiteration:ga("Animation","AnimationIteration"),animationstart:ga("Animation","AnimationStart"),transitionend:ga("Transition","TransitionEnd")},ps={},lm={};Jt&&(lm=document.createElement("div").style,"AnimationEvent"in window||(delete pr.animationend.animation,delete pr.animationiteration.animation,delete pr.animationstart.animation),"TransitionEvent"in window||delete pr.transitionend.transition);function Ao(e){if(ps[e])return ps[e];if(!pr[e])return e;var t=pr[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in lm)return ps[e]=t[n];return e}var um=Ao("animationend"),cm=Ao("animationiteration"),dm=Ao("animationstart"),pm=Ao("transitionend"),fm=new Map,ad="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function zn(e,t){fm.set(e,t),er(t,[e])}for(var fs=0;fs<ad.length;fs++){var ms=ad[fs],J0=ms.toLowerCase(),ex=ms[0].toUpperCase()+ms.slice(1);zn(J0,"on"+ex)}zn(um,"onAnimationEnd");zn(cm,"onAnimationIteration");zn(dm,"onAnimationStart");zn("dblclick","onDoubleClick");zn("focusin","onFocus");zn("focusout","onBlur");zn(pm,"onTransitionEnd");jr("onMouseEnter",["mouseout","mouseover"]);jr("onMouseLeave",["mouseout","mouseover"]);jr("onPointerEnter",["pointerout","pointerover"]);jr("onPointerLeave",["pointerout","pointerover"]);er("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));er("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));er("onBeforeInput",["compositionend","keypress","textInput","paste"]);er("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));er("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));er("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var pi="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),tx=new Set("cancel close invalid load scroll toggle".split(" ").concat(pi));function od(e,t,n){var r=e.type||"unknown-event";e.currentTarget=n,Jy(r,t,void 0,e),e.currentTarget=null}function mm(e,t){t=(t&4)!==0;for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;e:{var a=void 0;if(t)for(var o=r.length-1;0<=o;o--){var s=r[o],l=s.instance,u=s.currentTarget;if(s=s.listener,l!==a&&i.isPropagationStopped())break e;od(i,s,u),a=l}else for(o=0;o<r.length;o++){if(s=r[o],l=s.instance,u=s.currentTarget,s=s.listener,l!==a&&i.isPropagationStopped())break e;od(i,s,u),a=l}}}if(Ja)throw e=hl,Ja=!1,hl=null,e}function fe(e,t){var n=t[Cl];n===void 0&&(n=t[Cl]=new Set);var r=e+"__bubble";n.has(r)||(hm(t,e,2,!1),n.add(r))}function hs(e,t,n){var r=0;t&&(r|=4),hm(n,e,r,t)}var ya="_reactListening"+Math.random().toString(36).slice(2);function Ii(e){if(!e[ya]){e[ya]=!0,wf.forEach(function(n){n!=="selectionchange"&&(tx.has(n)||hs(n,!1,e),hs(n,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[ya]||(t[ya]=!0,hs("selectionchange",!1,t))}}function hm(e,t,n,r){switch(Zf(t)){case 1:var i=h0;break;case 4:i=g0;break;default:i=Tu}n=i.bind(null,t,n,e),i=void 0,!ml||t!=="touchstart"&&t!=="touchmove"&&t!=="wheel"||(i=!0),r?i!==void 0?e.addEventListener(t,n,{capture:!0,passive:i}):e.addEventListener(t,n,!0):i!==void 0?e.addEventListener(t,n,{passive:i}):e.addEventListener(t,n,!1)}function gs(e,t,n,r,i){var a=r;if(!(t&1)&&!(t&2)&&r!==null)e:for(;;){if(r===null)return;var o=r.tag;if(o===3||o===4){var s=r.stateNode.containerInfo;if(s===i||s.nodeType===8&&s.parentNode===i)break;if(o===4)for(o=r.return;o!==null;){var l=o.tag;if((l===3||l===4)&&(l=o.stateNode.containerInfo,l===i||l.nodeType===8&&l.parentNode===i))return;o=o.return}for(;s!==null;){if(o=Wn(s),o===null)return;if(l=o.tag,l===5||l===6){r=a=o;continue e}s=s.parentNode}}r=r.return}Ff(function(){var u=a,d=Su(n),c=[];e:{var f=fm.get(e);if(f!==void 0){var m=Nu,y=e;switch(e){case"keypress":if(Fa(n)===0)break e;case"keydown":case"keyup":m=j0;break;case"focusin":y="focus",m=us;break;case"focusout":y="blur",m=us;break;case"beforeblur":case"afterblur":m=us;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":m=Kc;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":m=_0;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":m=R0;break;case um:case cm:case dm:m=w0;break;case pm:m=O0;break;case"scroll":m=y0;break;case"wheel":m=F0;break;case"copy":case"cut":case"paste":m=S0;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":m=Xc}var _=(t&4)!==0,k=!_&&e==="scroll",h=_?f!==null?f+"Capture":null:f;_=[];for(var g=u,x;g!==null;){x=g;var C=x.stateNode;if(x.tag===5&&C!==null&&(x=C,h!==null&&(C=Pi(g,h),C!=null&&_.push(Oi(g,C,x)))),k)break;g=g.return}0<_.length&&(f=new m(f,y,null,n,d),c.push({event:f,listeners:_}))}}if(!(t&7)){e:{if(f=e==="mouseover"||e==="pointerover",m=e==="mouseout"||e==="pointerout",f&&n!==pl&&(y=n.relatedTarget||n.fromElement)&&(Wn(y)||y[en]))break e;if((m||f)&&(f=d.window===d?d:(f=d.ownerDocument)?f.defaultView||f.parentWindow:window,m?(y=n.relatedTarget||n.toElement,m=u,y=y?Wn(y):null,y!==null&&(k=tr(y),y!==k||y.tag!==5&&y.tag!==6)&&(y=null)):(m=null,y=u),m!==y)){if(_=Kc,C="onMouseLeave",h="onMouseEnter",g="mouse",(e==="pointerout"||e==="pointerover")&&(_=Xc,C="onPointerLeave",h="onPointerEnter",g="pointer"),k=m==null?f:fr(m),x=y==null?f:fr(y),f=new _(C,g+"leave",m,n,d),f.target=k,f.relatedTarget=x,C=null,Wn(d)===u&&(_=new _(h,g+"enter",y,n,d),_.target=x,_.relatedTarget=k,C=_),k=C,m&&y)t:{for(_=m,h=y,g=0,x=_;x;x=ir(x))g++;for(x=0,C=h;C;C=ir(C))x++;for(;0<g-x;)_=ir(_),g--;for(;0<x-g;)h=ir(h),x--;for(;g--;){if(_===h||h!==null&&_===h.alternate)break t;_=ir(_),h=ir(h)}_=null}else _=null;m!==null&&sd(c,f,m,_,!1),y!==null&&k!==null&&sd(c,k,y,_,!0)}}e:{if(f=u?fr(u):window,m=f.nodeName&&f.nodeName.toLowerCase(),m==="select"||m==="input"&&f.type==="file")var z=H0;else if(Jc(f))if(im)z=Y0;else{z=Q0;var S=$0}else(m=f.nodeName)&&m.toLowerCase()==="input"&&(f.type==="checkbox"||f.type==="radio")&&(z=K0);if(z&&(z=z(e,u))){rm(c,z,n,d);break e}S&&S(e,f,u),e==="focusout"&&(S=f._wrapperState)&&S.controlled&&f.type==="number"&&sl(f,"number",f.value)}switch(S=u?fr(u):window,e){case"focusin":(Jc(S)||S.contentEditable==="true")&&(dr=S,_l=u,xi=null);break;case"focusout":xi=_l=dr=null;break;case"mousedown":vl=!0;break;case"contextmenu":case"mouseup":case"dragend":vl=!1,id(c,n,d);break;case"selectionchange":if(Z0)break;case"keydown":case"keyup":id(c,n,d)}var T;if(Lu)e:{switch(e){case"compositionstart":var N="onCompositionStart";break e;case"compositionend":N="onCompositionEnd";break e;case"compositionupdate":N="onCompositionUpdate";break e}N=void 0}else cr?tm(e,n)&&(N="onCompositionEnd"):e==="keydown"&&n.keyCode===229&&(N="onCompositionStart");N&&(em&&n.locale!=="ko"&&(cr||N!=="onCompositionStart"?N==="onCompositionEnd"&&cr&&(T=Jf()):(pn=d,Pu="value"in pn?pn.value:pn.textContent,cr=!0)),S=io(u,N),0<S.length&&(N=new Yc(N,e,null,n,d),c.push({event:N,listeners:S}),T?N.data=T:(T=nm(n),T!==null&&(N.data=T)))),(T=q0?W0(e,n):B0(e,n))&&(u=io(u,"onBeforeInput"),0<u.length&&(d=new Yc("onBeforeInput","beforeinput",null,n,d),c.push({event:d,listeners:u}),d.data=T))}mm(c,t)})}function Oi(e,t,n){return{instance:e,listener:t,currentTarget:n}}function io(e,t){for(var n=t+"Capture",r=[];e!==null;){var i=e,a=i.stateNode;i.tag===5&&a!==null&&(i=a,a=Pi(e,n),a!=null&&r.unshift(Oi(e,a,i)),a=Pi(e,t),a!=null&&r.push(Oi(e,a,i))),e=e.return}return r}function ir(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function sd(e,t,n,r,i){for(var a=t._reactName,o=[];n!==null&&n!==r;){var s=n,l=s.alternate,u=s.stateNode;if(l!==null&&l===r)break;s.tag===5&&u!==null&&(s=u,i?(l=Pi(n,a),l!=null&&o.unshift(Oi(n,l,s))):i||(l=Pi(n,a),l!=null&&o.push(Oi(n,l,s)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var nx=/\r\n?/g,rx=/\u0000|\uFFFD/g;function ld(e){return(typeof e=="string"?e:""+e).replace(nx,`
`).replace(rx,"")}function xa(e,t,n){if(t=ld(t),ld(e)!==t&&n)throw Error(A(425))}function ao(){}var bl=null,wl=null;function kl(e,t){return e==="textarea"||e==="noscript"||typeof t.children=="string"||typeof t.children=="number"||typeof t.dangerouslySetInnerHTML=="object"&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var Sl=typeof setTimeout=="function"?setTimeout:void 0,ix=typeof clearTimeout=="function"?clearTimeout:void 0,ud=typeof Promise=="function"?Promise:void 0,ax=typeof queueMicrotask=="function"?queueMicrotask:typeof ud<"u"?function(e){return ud.resolve(null).then(e).catch(ox)}:Sl;function ox(e){setTimeout(function(){throw e})}function ys(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8)if(n=i.data,n==="/$"){if(r===0){e.removeChild(i),Li(t);return}r--}else n!=="$"&&n!=="$?"&&n!=="$!"||r++;n=i}while(n);Li(t)}function xn(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t==="$"||t==="$!"||t==="$?")break;if(t==="/$")return null}}return e}function cd(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="$"||n==="$!"||n==="$?"){if(t===0)return e;t--}else n==="/$"&&t++}e=e.previousSibling}return null}var Vr=Math.random().toString(36).slice(2),qt="__reactFiber$"+Vr,Di="__reactProps$"+Vr,en="__reactContainer$"+Vr,Cl="__reactEvents$"+Vr,sx="__reactListeners$"+Vr,lx="__reactHandles$"+Vr;function Wn(e){var t=e[qt];if(t)return t;for(var n=e.parentNode;n;){if(t=n[en]||n[qt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=cd(e);e!==null;){if(n=e[qt])return n;e=cd(e)}return t}e=n,n=e.parentNode}return null}function Ji(e){return e=e[qt]||e[en],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function fr(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(A(33))}function Ro(e){return e[Di]||null}var El=[],mr=-1;function Tn(e){return{current:e}}function me(e){0>mr||(e.current=El[mr],El[mr]=null,mr--)}function pe(e,t){mr++,El[mr]=e.current,e.current=t}var En={},qe=Tn(En),Ge=Tn(!1),Qn=En;function Lr(e,t){var n=e.type.contextTypes;if(!n)return En;var r=e.stateNode;if(r&&r.__reactInternalMemoizedUnmaskedChildContext===t)return r.__reactInternalMemoizedMaskedChildContext;var i={},a;for(a in n)i[a]=t[a];return r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=t,e.__reactInternalMemoizedMaskedChildContext=i),i}function Ze(e){return e=e.childContextTypes,e!=null}function oo(){me(Ge),me(qe)}function dd(e,t,n){if(qe.current!==En)throw Error(A(168));pe(qe,t),pe(Ge,n)}function gm(e,t,n){var r=e.stateNode;if(t=t.childContextTypes,typeof r.getChildContext!="function")return n;r=r.getChildContext();for(var i in r)if(!(i in t))throw Error(A(108,$y(e)||"Unknown",i));return xe({},n,r)}function so(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||En,Qn=qe.current,pe(qe,e),pe(Ge,Ge.current),!0}function pd(e,t,n){var r=e.stateNode;if(!r)throw Error(A(169));n?(e=gm(e,t,Qn),r.__reactInternalMemoizedMergedChildContext=e,me(Ge),me(qe),pe(qe,e)):me(Ge),pe(Ge,n)}var Yt=null,Io=!1,xs=!1;function ym(e){Yt===null?Yt=[e]:Yt.push(e)}function ux(e){Io=!0,ym(e)}function Pn(){if(!xs&&Yt!==null){xs=!0;var e=0,t=se;try{var n=Yt;for(se=1;e<n.length;e++){var r=n[e];do r=r(!0);while(r!==null)}Yt=null,Io=!1}catch(i){throw Yt!==null&&(Yt=Yt.slice(e+1)),Bf(Cu,Pn),i}finally{se=t,xs=!1}}return null}var hr=[],gr=0,lo=null,uo=0,ht=[],gt=0,Kn=null,Xt=1,Gt="";function Dn(e,t){hr[gr++]=uo,hr[gr++]=lo,lo=e,uo=t}function xm(e,t,n){ht[gt++]=Xt,ht[gt++]=Gt,ht[gt++]=Kn,Kn=e;var r=Xt;e=Gt;var i=32-Lt(r)-1;r&=~(1<<i),n+=1;var a=32-Lt(t)+i;if(30<a){var o=i-i%5;a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,Xt=1<<32-Lt(t)+i|n<<i|r,Gt=a+e}else Xt=1<<a|n<<i|r,Gt=e}function Ru(e){e.return!==null&&(Dn(e,1),xm(e,1,0))}function Iu(e){for(;e===lo;)lo=hr[--gr],hr[gr]=null,uo=hr[--gr],hr[gr]=null;for(;e===Kn;)Kn=ht[--gt],ht[gt]=null,Gt=ht[--gt],ht[gt]=null,Xt=ht[--gt],ht[gt]=null}var st=null,at=null,he=!1,Nt=null;function _m(e,t){var n=xt(5,null,null,0);n.elementType="DELETED",n.stateNode=t,n.return=e,t=e.deletions,t===null?(e.deletions=[n],e.flags|=16):t.push(n)}function fd(e,t){switch(e.tag){case 5:var n=e.type;return t=t.nodeType!==1||n.toLowerCase()!==t.nodeName.toLowerCase()?null:t,t!==null?(e.stateNode=t,st=e,at=xn(t.firstChild),!0):!1;case 6:return t=e.pendingProps===""||t.nodeType!==3?null:t,t!==null?(e.stateNode=t,st=e,at=null,!0):!1;case 13:return t=t.nodeType!==8?null:t,t!==null?(n=Kn!==null?{id:Xt,overflow:Gt}:null,e.memoizedState={dehydrated:t,treeContext:n,retryLane:1073741824},n=xt(18,null,null,0),n.stateNode=t,n.return=e,e.child=n,st=e,at=null,!0):!1;default:return!1}}function zl(e){return(e.mode&1)!==0&&(e.flags&128)===0}function Tl(e){if(he){var t=at;if(t){var n=t;if(!fd(e,t)){if(zl(e))throw Error(A(418));t=xn(n.nextSibling);var r=st;t&&fd(e,t)?_m(r,n):(e.flags=e.flags&-4097|2,he=!1,st=e)}}else{if(zl(e))throw Error(A(418));e.flags=e.flags&-4097|2,he=!1,st=e}}}function md(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;st=e}function _a(e){if(e!==st)return!1;if(!he)return md(e),he=!0,!1;var t;if((t=e.tag!==3)&&!(t=e.tag!==5)&&(t=e.type,t=t!=="head"&&t!=="body"&&!kl(e.type,e.memoizedProps)),t&&(t=at)){if(zl(e))throw vm(),Error(A(418));for(;t;)_m(e,t),t=xn(t.nextSibling)}if(md(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(A(317));e:{for(e=e.nextSibling,t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="/$"){if(t===0){at=xn(e.nextSibling);break e}t--}else n!=="$"&&n!=="$!"&&n!=="$?"||t++}e=e.nextSibling}at=null}}else at=st?xn(e.stateNode.nextSibling):null;return!0}function vm(){for(var e=at;e;)e=xn(e.nextSibling)}function Ar(){at=st=null,he=!1}function Ou(e){Nt===null?Nt=[e]:Nt.push(e)}var cx=rn.ReactCurrentBatchConfig;function ti(e,t,n){if(e=n.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(A(309));var r=n.stateNode}if(!r)throw Error(A(147,e));var i=r,a=""+e;return t!==null&&t.ref!==null&&typeof t.ref=="function"&&t.ref._stringRef===a?t.ref:(t=function(o){var s=i.refs;o===null?delete s[a]:s[a]=o},t._stringRef=a,t)}if(typeof e!="string")throw Error(A(284));if(!n._owner)throw Error(A(290,e))}return e}function va(e,t){throw e=Object.prototype.toString.call(t),Error(A(31,e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e))}function hd(e){var t=e._init;return t(e._payload)}function bm(e){function t(h,g){if(e){var x=h.deletions;x===null?(h.deletions=[g],h.flags|=16):x.push(g)}}function n(h,g){if(!e)return null;for(;g!==null;)t(h,g),g=g.sibling;return null}function r(h,g){for(h=new Map;g!==null;)g.key!==null?h.set(g.key,g):h.set(g.index,g),g=g.sibling;return h}function i(h,g){return h=wn(h,g),h.index=0,h.sibling=null,h}function a(h,g,x){return h.index=x,e?(x=h.alternate,x!==null?(x=x.index,x<g?(h.flags|=2,g):x):(h.flags|=2,g)):(h.flags|=1048576,g)}function o(h){return e&&h.alternate===null&&(h.flags|=2),h}function s(h,g,x,C){return g===null||g.tag!==6?(g=Cs(x,h.mode,C),g.return=h,g):(g=i(g,x),g.return=h,g)}function l(h,g,x,C){var z=x.type;return z===ur?d(h,g,x.props.children,C,x.key):g!==null&&(g.elementType===z||typeof z=="object"&&z!==null&&z.$$typeof===ln&&hd(z)===g.type)?(C=i(g,x.props),C.ref=ti(h,g,x),C.return=h,C):(C=Ha(x.type,x.key,x.props,null,h.mode,C),C.ref=ti(h,g,x),C.return=h,C)}function u(h,g,x,C){return g===null||g.tag!==4||g.stateNode.containerInfo!==x.containerInfo||g.stateNode.implementation!==x.implementation?(g=Es(x,h.mode,C),g.return=h,g):(g=i(g,x.children||[]),g.return=h,g)}function d(h,g,x,C,z){return g===null||g.tag!==7?(g=$n(x,h.mode,C,z),g.return=h,g):(g=i(g,x),g.return=h,g)}function c(h,g,x){if(typeof g=="string"&&g!==""||typeof g=="number")return g=Cs(""+g,h.mode,x),g.return=h,g;if(typeof g=="object"&&g!==null){switch(g.$$typeof){case ua:return x=Ha(g.type,g.key,g.props,null,h.mode,x),x.ref=ti(h,null,g),x.return=h,x;case lr:return g=Es(g,h.mode,x),g.return=h,g;case ln:var C=g._init;return c(h,C(g._payload),x)}if(ci(g)||Xr(g))return g=$n(g,h.mode,x,null),g.return=h,g;va(h,g)}return null}function f(h,g,x,C){var z=g!==null?g.key:null;if(typeof x=="string"&&x!==""||typeof x=="number")return z!==null?null:s(h,g,""+x,C);if(typeof x=="object"&&x!==null){switch(x.$$typeof){case ua:return x.key===z?l(h,g,x,C):null;case lr:return x.key===z?u(h,g,x,C):null;case ln:return z=x._init,f(h,g,z(x._payload),C)}if(ci(x)||Xr(x))return z!==null?null:d(h,g,x,C,null);va(h,x)}return null}function m(h,g,x,C,z){if(typeof C=="string"&&C!==""||typeof C=="number")return h=h.get(x)||null,s(g,h,""+C,z);if(typeof C=="object"&&C!==null){switch(C.$$typeof){case ua:return h=h.get(C.key===null?x:C.key)||null,l(g,h,C,z);case lr:return h=h.get(C.key===null?x:C.key)||null,u(g,h,C,z);case ln:var S=C._init;return m(h,g,x,S(C._payload),z)}if(ci(C)||Xr(C))return h=h.get(x)||null,d(g,h,C,z,null);va(g,C)}return null}function y(h,g,x,C){for(var z=null,S=null,T=g,N=g=0,M=null;T!==null&&N<x.length;N++){T.index>N?(M=T,T=null):M=T.sibling;var w=f(h,T,x[N],C);if(w===null){T===null&&(T=M);break}e&&T&&w.alternate===null&&t(h,T),g=a(w,g,N),S===null?z=w:S.sibling=w,S=w,T=M}if(N===x.length)return n(h,T),he&&Dn(h,N),z;if(T===null){for(;N<x.length;N++)T=c(h,x[N],C),T!==null&&(g=a(T,g,N),S===null?z=T:S.sibling=T,S=T);return he&&Dn(h,N),z}for(T=r(h,T);N<x.length;N++)M=m(T,h,N,x[N],C),M!==null&&(e&&M.alternate!==null&&T.delete(M.key===null?N:M.key),g=a(M,g,N),S===null?z=M:S.sibling=M,S=M);return e&&T.forEach(function(I){return t(h,I)}),he&&Dn(h,N),z}function _(h,g,x,C){var z=Xr(x);if(typeof z!="function")throw Error(A(150));if(x=z.call(x),x==null)throw Error(A(151));for(var S=z=null,T=g,N=g=0,M=null,w=x.next();T!==null&&!w.done;N++,w=x.next()){T.index>N?(M=T,T=null):M=T.sibling;var I=f(h,T,w.value,C);if(I===null){T===null&&(T=M);break}e&&T&&I.alternate===null&&t(h,T),g=a(I,g,N),S===null?z=I:S.sibling=I,S=I,T=M}if(w.done)return n(h,T),he&&Dn(h,N),z;if(T===null){for(;!w.done;N++,w=x.next())w=c(h,w.value,C),w!==null&&(g=a(w,g,N),S===null?z=w:S.sibling=w,S=w);return he&&Dn(h,N),z}for(T=r(h,T);!w.done;N++,w=x.next())w=m(T,h,N,w.value,C),w!==null&&(e&&w.alternate!==null&&T.delete(w.key===null?N:w.key),g=a(w,g,N),S===null?z=w:S.sibling=w,S=w);return e&&T.forEach(function(W){return t(h,W)}),he&&Dn(h,N),z}function k(h,g,x,C){if(typeof x=="object"&&x!==null&&x.type===ur&&x.key===null&&(x=x.props.children),typeof x=="object"&&x!==null){switch(x.$$typeof){case ua:e:{for(var z=x.key,S=g;S!==null;){if(S.key===z){if(z=x.type,z===ur){if(S.tag===7){n(h,S.sibling),g=i(S,x.props.children),g.return=h,h=g;break e}}else if(S.elementType===z||typeof z=="object"&&z!==null&&z.$$typeof===ln&&hd(z)===S.type){n(h,S.sibling),g=i(S,x.props),g.ref=ti(h,S,x),g.return=h,h=g;break e}n(h,S);break}else t(h,S);S=S.sibling}x.type===ur?(g=$n(x.props.children,h.mode,C,x.key),g.return=h,h=g):(C=Ha(x.type,x.key,x.props,null,h.mode,C),C.ref=ti(h,g,x),C.return=h,h=C)}return o(h);case lr:e:{for(S=x.key;g!==null;){if(g.key===S)if(g.tag===4&&g.stateNode.containerInfo===x.containerInfo&&g.stateNode.implementation===x.implementation){n(h,g.sibling),g=i(g,x.children||[]),g.return=h,h=g;break e}else{n(h,g);break}else t(h,g);g=g.sibling}g=Es(x,h.mode,C),g.return=h,h=g}return o(h);case ln:return S=x._init,k(h,g,S(x._payload),C)}if(ci(x))return y(h,g,x,C);if(Xr(x))return _(h,g,x,C);va(h,x)}return typeof x=="string"&&x!==""||typeof x=="number"?(x=""+x,g!==null&&g.tag===6?(n(h,g.sibling),g=i(g,x),g.return=h,h=g):(n(h,g),g=Cs(x,h.mode,C),g.return=h,h=g),o(h)):n(h,g)}return k}var Rr=bm(!0),wm=bm(!1),co=Tn(null),po=null,yr=null,Du=null;function Fu(){Du=yr=po=null}function Mu(e){var t=co.current;me(co),e._currentValue=t}function Pl(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)!==t?(e.childLanes|=t,r!==null&&(r.childLanes|=t)):r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t),e===n)break;e=e.return}}function Cr(e,t){po=e,Du=yr=null,e=e.dependencies,e!==null&&e.firstContext!==null&&(e.lanes&t&&(Xe=!0),e.firstContext=null)}function vt(e){var t=e._currentValue;if(Du!==e)if(e={context:e,memoizedValue:t,next:null},yr===null){if(po===null)throw Error(A(308));yr=e,po.dependencies={lanes:0,firstContext:e}}else yr=yr.next=e;return t}var Bn=null;function qu(e){Bn===null?Bn=[e]:Bn.push(e)}function km(e,t,n,r){var i=t.interleaved;return i===null?(n.next=n,qu(t)):(n.next=i.next,i.next=n),t.interleaved=n,tn(e,r)}function tn(e,t){e.lanes|=t;var n=e.alternate;for(n!==null&&(n.lanes|=t),n=e,e=e.return;e!==null;)e.childLanes|=t,n=e.alternate,n!==null&&(n.childLanes|=t),n=e,e=e.return;return n.tag===3?n.stateNode:null}var un=!1;function Wu(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function Sm(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function Zt(e,t){return{eventTime:e,lane:t,tag:0,payload:null,callback:null,next:null}}function _n(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,ae&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,tn(e,n)}return i=r.interleaved,i===null?(t.next=t,qu(r)):(t.next=i.next,i.next=t),r.interleaved=t,tn(e,n)}function Ma(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,(n&4194240)!==0)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Eu(e,n)}}function gd(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,a=null;if(n=n.firstBaseUpdate,n!==null){do{var o={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};a===null?i=a=o:a=a.next=o,n=n.next}while(n!==null);a===null?i=a=t:a=a.next=t}else i=a=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:a,shared:r.shared,effects:r.effects},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}function fo(e,t,n,r){var i=e.updateQueue;un=!1;var a=i.firstBaseUpdate,o=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var l=s,u=l.next;l.next=null,o===null?a=u:o.next=u,o=l;var d=e.alternate;d!==null&&(d=d.updateQueue,s=d.lastBaseUpdate,s!==o&&(s===null?d.firstBaseUpdate=u:s.next=u,d.lastBaseUpdate=l))}if(a!==null){var c=i.baseState;o=0,d=u=l=null,s=a;do{var f=s.lane,m=s.eventTime;if((r&f)===f){d!==null&&(d=d.next={eventTime:m,lane:0,tag:s.tag,payload:s.payload,callback:s.callback,next:null});e:{var y=e,_=s;switch(f=t,m=n,_.tag){case 1:if(y=_.payload,typeof y=="function"){c=y.call(m,c,f);break e}c=y;break e;case 3:y.flags=y.flags&-65537|128;case 0:if(y=_.payload,f=typeof y=="function"?y.call(m,c,f):y,f==null)break e;c=xe({},c,f);break e;case 2:un=!0}}s.callback!==null&&s.lane!==0&&(e.flags|=64,f=i.effects,f===null?i.effects=[s]:f.push(s))}else m={eventTime:m,lane:f,tag:s.tag,payload:s.payload,callback:s.callback,next:null},d===null?(u=d=m,l=c):d=d.next=m,o|=f;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;f=s,s=f.next,f.next=null,i.lastBaseUpdate=f,i.shared.pending=null}}while(!0);if(d===null&&(l=c),i.baseState=l,i.firstBaseUpdate=u,i.lastBaseUpdate=d,t=i.shared.interleaved,t!==null){i=t;do o|=i.lane,i=i.next;while(i!==t)}else a===null&&(i.shared.lanes=0);Xn|=o,e.lanes=o,e.memoizedState=c}}function yd(e,t,n){if(e=t.effects,t.effects=null,e!==null)for(t=0;t<e.length;t++){var r=e[t],i=r.callback;if(i!==null){if(r.callback=null,r=n,typeof i!="function")throw Error(A(191,i));i.call(r)}}}var ea={},Bt=Tn(ea),Fi=Tn(ea),Mi=Tn(ea);function Vn(e){if(e===ea)throw Error(A(174));return e}function Bu(e,t){switch(pe(Mi,t),pe(Fi,e),pe(Bt,ea),e=t.nodeType,e){case 9:case 11:t=(t=t.documentElement)?t.namespaceURI:ul(null,"");break;default:e=e===8?t.parentNode:t,t=e.namespaceURI||null,e=e.tagName,t=ul(t,e)}me(Bt),pe(Bt,t)}function Ir(){me(Bt),me(Fi),me(Mi)}function Cm(e){Vn(Mi.current);var t=Vn(Bt.current),n=ul(t,e.type);t!==n&&(pe(Fi,e),pe(Bt,n))}function Vu(e){Fi.current===e&&(me(Bt),me(Fi))}var ge=Tn(0);function mo(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==void 0){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var _s=[];function Uu(){for(var e=0;e<_s.length;e++)_s[e]._workInProgressVersionPrimary=null;_s.length=0}var qa=rn.ReactCurrentDispatcher,vs=rn.ReactCurrentBatchConfig,Yn=0,ye=null,ze=null,Pe=null,ho=!1,_i=!1,qi=0,dx=0;function Ie(){throw Error(A(321))}function Hu(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!It(e[n],t[n]))return!1;return!0}function $u(e,t,n,r,i,a){if(Yn=a,ye=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,qa.current=e===null||e.memoizedState===null?hx:gx,e=n(r,i),_i){a=0;do{if(_i=!1,qi=0,25<=a)throw Error(A(301));a+=1,Pe=ze=null,t.updateQueue=null,qa.current=yx,e=n(r,i)}while(_i)}if(qa.current=go,t=ze!==null&&ze.next!==null,Yn=0,Pe=ze=ye=null,ho=!1,t)throw Error(A(300));return e}function Qu(){var e=qi!==0;return qi=0,e}function Ft(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Pe===null?ye.memoizedState=Pe=e:Pe=Pe.next=e,Pe}function bt(){if(ze===null){var e=ye.alternate;e=e!==null?e.memoizedState:null}else e=ze.next;var t=Pe===null?ye.memoizedState:Pe.next;if(t!==null)Pe=t,ze=e;else{if(e===null)throw Error(A(310));ze=e,e={memoizedState:ze.memoizedState,baseState:ze.baseState,baseQueue:ze.baseQueue,queue:ze.queue,next:null},Pe===null?ye.memoizedState=Pe=e:Pe=Pe.next=e}return Pe}function Wi(e,t){return typeof t=="function"?t(e):t}function bs(e){var t=bt(),n=t.queue;if(n===null)throw Error(A(311));n.lastRenderedReducer=e;var r=ze,i=r.baseQueue,a=n.pending;if(a!==null){if(i!==null){var o=i.next;i.next=a.next,a.next=o}r.baseQueue=i=a,n.pending=null}if(i!==null){a=i.next,r=r.baseState;var s=o=null,l=null,u=a;do{var d=u.lane;if((Yn&d)===d)l!==null&&(l=l.next={lane:0,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),r=u.hasEagerState?u.eagerState:e(r,u.action);else{var c={lane:d,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null};l===null?(s=l=c,o=r):l=l.next=c,ye.lanes|=d,Xn|=d}u=u.next}while(u!==null&&u!==a);l===null?o=r:l.next=s,It(r,t.memoizedState)||(Xe=!0),t.memoizedState=r,t.baseState=o,t.baseQueue=l,n.lastRenderedState=r}if(e=n.interleaved,e!==null){i=e;do a=i.lane,ye.lanes|=a,Xn|=a,i=i.next;while(i!==e)}else i===null&&(n.lanes=0);return[t.memoizedState,n.dispatch]}function ws(e){var t=bt(),n=t.queue;if(n===null)throw Error(A(311));n.lastRenderedReducer=e;var r=n.dispatch,i=n.pending,a=t.memoizedState;if(i!==null){n.pending=null;var o=i=i.next;do a=e(a,o.action),o=o.next;while(o!==i);It(a,t.memoizedState)||(Xe=!0),t.memoizedState=a,t.baseQueue===null&&(t.baseState=a),n.lastRenderedState=a}return[a,r]}function Em(){}function zm(e,t){var n=ye,r=bt(),i=t(),a=!It(r.memoizedState,i);if(a&&(r.memoizedState=i,Xe=!0),r=r.queue,Ku(Nm.bind(null,n,r,e),[e]),r.getSnapshot!==t||a||Pe!==null&&Pe.memoizedState.tag&1){if(n.flags|=2048,Bi(9,Pm.bind(null,n,r,i,t),void 0,null),je===null)throw Error(A(349));Yn&30||Tm(n,t,i)}return i}function Tm(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=ye.updateQueue,t===null?(t={lastEffect:null,stores:null},ye.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function Pm(e,t,n,r){t.value=n,t.getSnapshot=r,jm(t)&&Lm(e)}function Nm(e,t,n){return n(function(){jm(t)&&Lm(e)})}function jm(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!It(e,n)}catch{return!0}}function Lm(e){var t=tn(e,1);t!==null&&At(t,e,1,-1)}function xd(e){var t=Ft();return typeof e=="function"&&(e=e()),t.memoizedState=t.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:Wi,lastRenderedState:e},t.queue=e,e=e.dispatch=mx.bind(null,ye,e),[t.memoizedState,e]}function Bi(e,t,n,r){return e={tag:e,create:t,destroy:n,deps:r,next:null},t=ye.updateQueue,t===null?(t={lastEffect:null,stores:null},ye.updateQueue=t,t.lastEffect=e.next=e):(n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e)),e}function Am(){return bt().memoizedState}function Wa(e,t,n,r){var i=Ft();ye.flags|=e,i.memoizedState=Bi(1|t,n,void 0,r===void 0?null:r)}function Oo(e,t,n,r){var i=bt();r=r===void 0?null:r;var a=void 0;if(ze!==null){var o=ze.memoizedState;if(a=o.destroy,r!==null&&Hu(r,o.deps)){i.memoizedState=Bi(t,n,a,r);return}}ye.flags|=e,i.memoizedState=Bi(1|t,n,a,r)}function _d(e,t){return Wa(8390656,8,e,t)}function Ku(e,t){return Oo(2048,8,e,t)}function Rm(e,t){return Oo(4,2,e,t)}function Im(e,t){return Oo(4,4,e,t)}function Om(e,t){if(typeof t=="function")return e=e(),t(e),function(){t(null)};if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function Dm(e,t,n){return n=n!=null?n.concat([e]):null,Oo(4,4,Om.bind(null,t,e),n)}function Yu(){}function Fm(e,t){var n=bt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&Hu(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function Mm(e,t){var n=bt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&Hu(t,r[1])?r[0]:(e=e(),n.memoizedState=[e,t],e)}function qm(e,t,n){return Yn&21?(It(n,t)||(n=Hf(),ye.lanes|=n,Xn|=n,e.baseState=!0),t):(e.baseState&&(e.baseState=!1,Xe=!0),e.memoizedState=n)}function px(e,t){var n=se;se=n!==0&&4>n?n:4,e(!0);var r=vs.transition;vs.transition={};try{e(!1),t()}finally{se=n,vs.transition=r}}function Wm(){return bt().memoizedState}function fx(e,t,n){var r=bn(e);if(n={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null},Bm(e))Vm(t,n);else if(n=km(e,t,n,r),n!==null){var i=He();At(n,e,r,i),Um(n,t,r)}}function mx(e,t,n){var r=bn(e),i={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null};if(Bm(e))Vm(t,i);else{var a=e.alternate;if(e.lanes===0&&(a===null||a.lanes===0)&&(a=t.lastRenderedReducer,a!==null))try{var o=t.lastRenderedState,s=a(o,n);if(i.hasEagerState=!0,i.eagerState=s,It(s,o)){var l=t.interleaved;l===null?(i.next=i,qu(t)):(i.next=l.next,l.next=i),t.interleaved=i;return}}catch{}finally{}n=km(e,t,i,r),n!==null&&(i=He(),At(n,e,r,i),Um(n,t,r))}}function Bm(e){var t=e.alternate;return e===ye||t!==null&&t===ye}function Vm(e,t){_i=ho=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function Um(e,t,n){if(n&4194240){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Eu(e,n)}}var go={readContext:vt,useCallback:Ie,useContext:Ie,useEffect:Ie,useImperativeHandle:Ie,useInsertionEffect:Ie,useLayoutEffect:Ie,useMemo:Ie,useReducer:Ie,useRef:Ie,useState:Ie,useDebugValue:Ie,useDeferredValue:Ie,useTransition:Ie,useMutableSource:Ie,useSyncExternalStore:Ie,useId:Ie,unstable_isNewReconciler:!1},hx={readContext:vt,useCallback:function(e,t){return Ft().memoizedState=[e,t===void 0?null:t],e},useContext:vt,useEffect:_d,useImperativeHandle:function(e,t,n){return n=n!=null?n.concat([e]):null,Wa(4194308,4,Om.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Wa(4194308,4,e,t)},useInsertionEffect:function(e,t){return Wa(4,2,e,t)},useMemo:function(e,t){var n=Ft();return t=t===void 0?null:t,e=e(),n.memoizedState=[e,t],e},useReducer:function(e,t,n){var r=Ft();return t=n!==void 0?n(t):t,r.memoizedState=r.baseState=t,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:t},r.queue=e,e=e.dispatch=fx.bind(null,ye,e),[r.memoizedState,e]},useRef:function(e){var t=Ft();return e={current:e},t.memoizedState=e},useState:xd,useDebugValue:Yu,useDeferredValue:function(e){return Ft().memoizedState=e},useTransition:function(){var e=xd(!1),t=e[0];return e=px.bind(null,e[1]),Ft().memoizedState=e,[t,e]},useMutableSource:function(){},useSyncExternalStore:function(e,t,n){var r=ye,i=Ft();if(he){if(n===void 0)throw Error(A(407));n=n()}else{if(n=t(),je===null)throw Error(A(349));Yn&30||Tm(r,t,n)}i.memoizedState=n;var a={value:n,getSnapshot:t};return i.queue=a,_d(Nm.bind(null,r,a,e),[e]),r.flags|=2048,Bi(9,Pm.bind(null,r,a,n,t),void 0,null),n},useId:function(){var e=Ft(),t=je.identifierPrefix;if(he){var n=Gt,r=Xt;n=(r&~(1<<32-Lt(r)-1)).toString(32)+n,t=":"+t+"R"+n,n=qi++,0<n&&(t+="H"+n.toString(32)),t+=":"}else n=dx++,t=":"+t+"r"+n.toString(32)+":";return e.memoizedState=t},unstable_isNewReconciler:!1},gx={readContext:vt,useCallback:Fm,useContext:vt,useEffect:Ku,useImperativeHandle:Dm,useInsertionEffect:Rm,useLayoutEffect:Im,useMemo:Mm,useReducer:bs,useRef:Am,useState:function(){return bs(Wi)},useDebugValue:Yu,useDeferredValue:function(e){var t=bt();return qm(t,ze.memoizedState,e)},useTransition:function(){var e=bs(Wi)[0],t=bt().memoizedState;return[e,t]},useMutableSource:Em,useSyncExternalStore:zm,useId:Wm,unstable_isNewReconciler:!1},yx={readContext:vt,useCallback:Fm,useContext:vt,useEffect:Ku,useImperativeHandle:Dm,useInsertionEffect:Rm,useLayoutEffect:Im,useMemo:Mm,useReducer:ws,useRef:Am,useState:function(){return ws(Wi)},useDebugValue:Yu,useDeferredValue:function(e){var t=bt();return ze===null?t.memoizedState=e:qm(t,ze.memoizedState,e)},useTransition:function(){var e=ws(Wi)[0],t=bt().memoizedState;return[e,t]},useMutableSource:Em,useSyncExternalStore:zm,useId:Wm,unstable_isNewReconciler:!1};function Tt(e,t){if(e&&e.defaultProps){t=xe({},t),e=e.defaultProps;for(var n in e)t[n]===void 0&&(t[n]=e[n]);return t}return t}function Nl(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:xe({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var Do={isMounted:function(e){return(e=e._reactInternals)?tr(e)===e:!1},enqueueSetState:function(e,t,n){e=e._reactInternals;var r=He(),i=bn(e),a=Zt(r,i);a.payload=t,n!=null&&(a.callback=n),t=_n(e,a,i),t!==null&&(At(t,e,i,r),Ma(t,e,i))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=He(),i=bn(e),a=Zt(r,i);a.tag=1,a.payload=t,n!=null&&(a.callback=n),t=_n(e,a,i),t!==null&&(At(t,e,i,r),Ma(t,e,i))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=He(),r=bn(e),i=Zt(n,r);i.tag=2,t!=null&&(i.callback=t),t=_n(e,i,r),t!==null&&(At(t,e,r,n),Ma(t,e,r))}};function vd(e,t,n,r,i,a,o){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(r,a,o):t.prototype&&t.prototype.isPureReactComponent?!Ri(n,r)||!Ri(i,a):!0}function Hm(e,t,n){var r=!1,i=En,a=t.contextType;return typeof a=="object"&&a!==null?a=vt(a):(i=Ze(t)?Qn:qe.current,r=t.contextTypes,a=(r=r!=null)?Lr(e,i):En),t=new t(n,a),e.memoizedState=t.state!==null&&t.state!==void 0?t.state:null,t.updater=Do,e.stateNode=t,t._reactInternals=e,r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=i,e.__reactInternalMemoizedMaskedChildContext=a),t}function bd(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps=="function"&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps=="function"&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&Do.enqueueReplaceState(t,t.state,null)}function jl(e,t,n,r){var i=e.stateNode;i.props=n,i.state=e.memoizedState,i.refs={},Wu(e);var a=t.contextType;typeof a=="object"&&a!==null?i.context=vt(a):(a=Ze(t)?Qn:qe.current,i.context=Lr(e,a)),i.state=e.memoizedState,a=t.getDerivedStateFromProps,typeof a=="function"&&(Nl(e,t,a,n),i.state=e.memoizedState),typeof t.getDerivedStateFromProps=="function"||typeof i.getSnapshotBeforeUpdate=="function"||typeof i.UNSAFE_componentWillMount!="function"&&typeof i.componentWillMount!="function"||(t=i.state,typeof i.componentWillMount=="function"&&i.componentWillMount(),typeof i.UNSAFE_componentWillMount=="function"&&i.UNSAFE_componentWillMount(),t!==i.state&&Do.enqueueReplaceState(i,i.state,null),fo(e,n,i,r),i.state=e.memoizedState),typeof i.componentDidMount=="function"&&(e.flags|=4194308)}function Or(e,t){try{var n="",r=t;do n+=Hy(r),r=r.return;while(r);var i=n}catch(a){i=`
Error generating stack: `+a.message+`
`+a.stack}return{value:e,source:t,stack:i,digest:null}}function ks(e,t,n){return{value:e,source:null,stack:n??null,digest:t??null}}function Ll(e,t){try{console.error(t.value)}catch(n){setTimeout(function(){throw n})}}var xx=typeof WeakMap=="function"?WeakMap:Map;function $m(e,t,n){n=Zt(-1,n),n.tag=3,n.payload={element:null};var r=t.value;return n.callback=function(){xo||(xo=!0,Bl=r),Ll(e,t)},n}function Qm(e,t,n){n=Zt(-1,n),n.tag=3;var r=e.type.getDerivedStateFromError;if(typeof r=="function"){var i=t.value;n.payload=function(){return r(i)},n.callback=function(){Ll(e,t)}}var a=e.stateNode;return a!==null&&typeof a.componentDidCatch=="function"&&(n.callback=function(){Ll(e,t),typeof r!="function"&&(vn===null?vn=new Set([this]):vn.add(this));var o=t.stack;this.componentDidCatch(t.value,{componentStack:o!==null?o:""})}),n}function wd(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new xx;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(i.add(n),e=Lx.bind(null,e,t,n),t.then(e,e))}function kd(e){do{var t;if((t=e.tag===13)&&(t=e.memoizedState,t=t!==null?t.dehydrated!==null:!0),t)return e;e=e.return}while(e!==null);return null}function Sd(e,t,n,r,i){return e.mode&1?(e.flags|=65536,e.lanes=i,e):(e===t?e.flags|=65536:(e.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(t=Zt(-1,1),t.tag=2,_n(n,t,1))),n.lanes|=1),e)}var _x=rn.ReactCurrentOwner,Xe=!1;function Ve(e,t,n,r){t.child=e===null?wm(t,null,n,r):Rr(t,e.child,n,r)}function Cd(e,t,n,r,i){n=n.render;var a=t.ref;return Cr(t,i),r=$u(e,t,n,r,a,i),n=Qu(),e!==null&&!Xe?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,nn(e,t,i)):(he&&n&&Ru(t),t.flags|=1,Ve(e,t,r,i),t.child)}function Ed(e,t,n,r,i){if(e===null){var a=n.type;return typeof a=="function"&&!rc(a)&&a.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(t.tag=15,t.type=a,Km(e,t,a,r,i)):(e=Ha(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(a=e.child,!(e.lanes&i)){var o=a.memoizedProps;if(n=n.compare,n=n!==null?n:Ri,n(o,r)&&e.ref===t.ref)return nn(e,t,i)}return t.flags|=1,e=wn(a,r),e.ref=t.ref,e.return=t,t.child=e}function Km(e,t,n,r,i){if(e!==null){var a=e.memoizedProps;if(Ri(a,r)&&e.ref===t.ref)if(Xe=!1,t.pendingProps=r=a,(e.lanes&i)!==0)e.flags&131072&&(Xe=!0);else return t.lanes=e.lanes,nn(e,t,i)}return Al(e,t,n,r,i)}function Ym(e,t,n){var r=t.pendingProps,i=r.children,a=e!==null?e.memoizedState:null;if(r.mode==="hidden")if(!(t.mode&1))t.memoizedState={baseLanes:0,cachePool:null,transitions:null},pe(_r,rt),rt|=n;else{if(!(n&1073741824))return e=a!==null?a.baseLanes|n:n,t.lanes=t.childLanes=1073741824,t.memoizedState={baseLanes:e,cachePool:null,transitions:null},t.updateQueue=null,pe(_r,rt),rt|=e,null;t.memoizedState={baseLanes:0,cachePool:null,transitions:null},r=a!==null?a.baseLanes:n,pe(_r,rt),rt|=r}else a!==null?(r=a.baseLanes|n,t.memoizedState=null):r=n,pe(_r,rt),rt|=r;return Ve(e,t,i,n),t.child}function Xm(e,t){var n=t.ref;(e===null&&n!==null||e!==null&&e.ref!==n)&&(t.flags|=512,t.flags|=2097152)}function Al(e,t,n,r,i){var a=Ze(n)?Qn:qe.current;return a=Lr(t,a),Cr(t,i),n=$u(e,t,n,r,a,i),r=Qu(),e!==null&&!Xe?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,nn(e,t,i)):(he&&r&&Ru(t),t.flags|=1,Ve(e,t,n,i),t.child)}function zd(e,t,n,r,i){if(Ze(n)){var a=!0;so(t)}else a=!1;if(Cr(t,i),t.stateNode===null)Ba(e,t),Hm(t,n,r),jl(t,n,r,i),r=!0;else if(e===null){var o=t.stateNode,s=t.memoizedProps;o.props=s;var l=o.context,u=n.contextType;typeof u=="object"&&u!==null?u=vt(u):(u=Ze(n)?Qn:qe.current,u=Lr(t,u));var d=n.getDerivedStateFromProps,c=typeof d=="function"||typeof o.getSnapshotBeforeUpdate=="function";c||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(s!==r||l!==u)&&bd(t,o,r,u),un=!1;var f=t.memoizedState;o.state=f,fo(t,r,o,i),l=t.memoizedState,s!==r||f!==l||Ge.current||un?(typeof d=="function"&&(Nl(t,n,d,r),l=t.memoizedState),(s=un||vd(t,n,s,r,f,l,u))?(c||typeof o.UNSAFE_componentWillMount!="function"&&typeof o.componentWillMount!="function"||(typeof o.componentWillMount=="function"&&o.componentWillMount(),typeof o.UNSAFE_componentWillMount=="function"&&o.UNSAFE_componentWillMount()),typeof o.componentDidMount=="function"&&(t.flags|=4194308)):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),o.props=r,o.state=l,o.context=u,r=s):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),r=!1)}else{o=t.stateNode,Sm(e,t),s=t.memoizedProps,u=t.type===t.elementType?s:Tt(t.type,s),o.props=u,c=t.pendingProps,f=o.context,l=n.contextType,typeof l=="object"&&l!==null?l=vt(l):(l=Ze(n)?Qn:qe.current,l=Lr(t,l));var m=n.getDerivedStateFromProps;(d=typeof m=="function"||typeof o.getSnapshotBeforeUpdate=="function")||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(s!==c||f!==l)&&bd(t,o,r,l),un=!1,f=t.memoizedState,o.state=f,fo(t,r,o,i);var y=t.memoizedState;s!==c||f!==y||Ge.current||un?(typeof m=="function"&&(Nl(t,n,m,r),y=t.memoizedState),(u=un||vd(t,n,u,r,f,y,l)||!1)?(d||typeof o.UNSAFE_componentWillUpdate!="function"&&typeof o.componentWillUpdate!="function"||(typeof o.componentWillUpdate=="function"&&o.componentWillUpdate(r,y,l),typeof o.UNSAFE_componentWillUpdate=="function"&&o.UNSAFE_componentWillUpdate(r,y,l)),typeof o.componentDidUpdate=="function"&&(t.flags|=4),typeof o.getSnapshotBeforeUpdate=="function"&&(t.flags|=1024)):(typeof o.componentDidUpdate!="function"||s===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=y),o.props=r,o.state=y,o.context=l,r=u):(typeof o.componentDidUpdate!="function"||s===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),r=!1)}return Rl(e,t,n,r,a,i)}function Rl(e,t,n,r,i,a){Xm(e,t);var o=(t.flags&128)!==0;if(!r&&!o)return i&&pd(t,n,!1),nn(e,t,a);r=t.stateNode,_x.current=t;var s=o&&typeof n.getDerivedStateFromError!="function"?null:r.render();return t.flags|=1,e!==null&&o?(t.child=Rr(t,e.child,null,a),t.child=Rr(t,null,s,a)):Ve(e,t,s,a),t.memoizedState=r.state,i&&pd(t,n,!0),t.child}function Gm(e){var t=e.stateNode;t.pendingContext?dd(e,t.pendingContext,t.pendingContext!==t.context):t.context&&dd(e,t.context,!1),Bu(e,t.containerInfo)}function Td(e,t,n,r,i){return Ar(),Ou(i),t.flags|=256,Ve(e,t,n,r),t.child}var Il={dehydrated:null,treeContext:null,retryLane:0};function Ol(e){return{baseLanes:e,cachePool:null,transitions:null}}function Zm(e,t,n){var r=t.pendingProps,i=ge.current,a=!1,o=(t.flags&128)!==0,s;if((s=o)||(s=e!==null&&e.memoizedState===null?!1:(i&2)!==0),s?(a=!0,t.flags&=-129):(e===null||e.memoizedState!==null)&&(i|=1),pe(ge,i&1),e===null)return Tl(t),e=t.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?(t.mode&1?e.data==="$!"?t.lanes=8:t.lanes=1073741824:t.lanes=1,null):(o=r.children,e=r.fallback,a?(r=t.mode,a=t.child,o={mode:"hidden",children:o},!(r&1)&&a!==null?(a.childLanes=0,a.pendingProps=o):a=qo(o,r,0,null),e=$n(e,r,n,null),a.return=t,e.return=t,a.sibling=e,t.child=a,t.child.memoizedState=Ol(n),t.memoizedState=Il,e):Xu(t,o));if(i=e.memoizedState,i!==null&&(s=i.dehydrated,s!==null))return vx(e,t,o,r,s,i,n);if(a){a=r.fallback,o=t.mode,i=e.child,s=i.sibling;var l={mode:"hidden",children:r.children};return!(o&1)&&t.child!==i?(r=t.child,r.childLanes=0,r.pendingProps=l,t.deletions=null):(r=wn(i,l),r.subtreeFlags=i.subtreeFlags&14680064),s!==null?a=wn(s,a):(a=$n(a,o,n,null),a.flags|=2),a.return=t,r.return=t,r.sibling=a,t.child=r,r=a,a=t.child,o=e.child.memoizedState,o=o===null?Ol(n):{baseLanes:o.baseLanes|n,cachePool:null,transitions:o.transitions},a.memoizedState=o,a.childLanes=e.childLanes&~n,t.memoizedState=Il,r}return a=e.child,e=a.sibling,r=wn(a,{mode:"visible",children:r.children}),!(t.mode&1)&&(r.lanes=n),r.return=t,r.sibling=null,e!==null&&(n=t.deletions,n===null?(t.deletions=[e],t.flags|=16):n.push(e)),t.child=r,t.memoizedState=null,r}function Xu(e,t){return t=qo({mode:"visible",children:t},e.mode,0,null),t.return=e,e.child=t}function ba(e,t,n,r){return r!==null&&Ou(r),Rr(t,e.child,null,n),e=Xu(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function vx(e,t,n,r,i,a,o){if(n)return t.flags&256?(t.flags&=-257,r=ks(Error(A(422))),ba(e,t,o,r)):t.memoizedState!==null?(t.child=e.child,t.flags|=128,null):(a=r.fallback,i=t.mode,r=qo({mode:"visible",children:r.children},i,0,null),a=$n(a,i,o,null),a.flags|=2,r.return=t,a.return=t,r.sibling=a,t.child=r,t.mode&1&&Rr(t,e.child,null,o),t.child.memoizedState=Ol(o),t.memoizedState=Il,a);if(!(t.mode&1))return ba(e,t,o,null);if(i.data==="$!"){if(r=i.nextSibling&&i.nextSibling.dataset,r)var s=r.dgst;return r=s,a=Error(A(419)),r=ks(a,r,void 0),ba(e,t,o,r)}if(s=(o&e.childLanes)!==0,Xe||s){if(r=je,r!==null){switch(o&-o){case 4:i=2;break;case 16:i=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:i=32;break;case 536870912:i=268435456;break;default:i=0}i=i&(r.suspendedLanes|o)?0:i,i!==0&&i!==a.retryLane&&(a.retryLane=i,tn(e,i),At(r,e,i,-1))}return nc(),r=ks(Error(A(421))),ba(e,t,o,r)}return i.data==="$?"?(t.flags|=128,t.child=e.child,t=Ax.bind(null,e),i._reactRetry=t,null):(e=a.treeContext,at=xn(i.nextSibling),st=t,he=!0,Nt=null,e!==null&&(ht[gt++]=Xt,ht[gt++]=Gt,ht[gt++]=Kn,Xt=e.id,Gt=e.overflow,Kn=t),t=Xu(t,r.children),t.flags|=4096,t)}function Pd(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),Pl(e.return,t,n)}function Ss(e,t,n,r,i){var a=e.memoizedState;a===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i}:(a.isBackwards=t,a.rendering=null,a.renderingStartTime=0,a.last=r,a.tail=n,a.tailMode=i)}function Jm(e,t,n){var r=t.pendingProps,i=r.revealOrder,a=r.tail;if(Ve(e,t,r.children,n),r=ge.current,r&2)r=r&1|2,t.flags|=128;else{if(e!==null&&e.flags&128)e:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&Pd(e,n,t);else if(e.tag===19)Pd(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}r&=1}if(pe(ge,r),!(t.mode&1))t.memoizedState=null;else switch(i){case"forwards":for(n=t.child,i=null;n!==null;)e=n.alternate,e!==null&&mo(e)===null&&(i=n),n=n.sibling;n=i,n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),Ss(t,!1,i,n,a);break;case"backwards":for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&mo(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}Ss(t,!0,n,null,a);break;case"together":Ss(t,!1,null,null,void 0);break;default:t.memoizedState=null}return t.child}function Ba(e,t){!(t.mode&1)&&e!==null&&(e.alternate=null,t.alternate=null,t.flags|=2)}function nn(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),Xn|=t.lanes,!(n&t.childLanes))return null;if(e!==null&&t.child!==e.child)throw Error(A(153));if(t.child!==null){for(e=t.child,n=wn(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=wn(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function bx(e,t,n){switch(t.tag){case 3:Gm(t),Ar();break;case 5:Cm(t);break;case 1:Ze(t.type)&&so(t);break;case 4:Bu(t,t.stateNode.containerInfo);break;case 10:var r=t.type._context,i=t.memoizedProps.value;pe(co,r._currentValue),r._currentValue=i;break;case 13:if(r=t.memoizedState,r!==null)return r.dehydrated!==null?(pe(ge,ge.current&1),t.flags|=128,null):n&t.child.childLanes?Zm(e,t,n):(pe(ge,ge.current&1),e=nn(e,t,n),e!==null?e.sibling:null);pe(ge,ge.current&1);break;case 19:if(r=(n&t.childLanes)!==0,e.flags&128){if(r)return Jm(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),pe(ge,ge.current),r)break;return null;case 22:case 23:return t.lanes=0,Ym(e,t,n)}return nn(e,t,n)}var eh,Dl,th,nh;eh=function(e,t){for(var n=t.child;n!==null;){if(n.tag===5||n.tag===6)e.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===t)break;for(;n.sibling===null;){if(n.return===null||n.return===t)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};Dl=function(){};th=function(e,t,n,r){var i=e.memoizedProps;if(i!==r){e=t.stateNode,Vn(Bt.current);var a=null;switch(n){case"input":i=al(e,i),r=al(e,r),a=[];break;case"select":i=xe({},i,{value:void 0}),r=xe({},r,{value:void 0}),a=[];break;case"textarea":i=ll(e,i),r=ll(e,r),a=[];break;default:typeof i.onClick!="function"&&typeof r.onClick=="function"&&(e.onclick=ao)}cl(n,r);var o;n=null;for(u in i)if(!r.hasOwnProperty(u)&&i.hasOwnProperty(u)&&i[u]!=null)if(u==="style"){var s=i[u];for(o in s)s.hasOwnProperty(o)&&(n||(n={}),n[o]="")}else u!=="dangerouslySetInnerHTML"&&u!=="children"&&u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&u!=="autoFocus"&&(zi.hasOwnProperty(u)?a||(a=[]):(a=a||[]).push(u,null));for(u in r){var l=r[u];if(s=i!=null?i[u]:void 0,r.hasOwnProperty(u)&&l!==s&&(l!=null||s!=null))if(u==="style")if(s){for(o in s)!s.hasOwnProperty(o)||l&&l.hasOwnProperty(o)||(n||(n={}),n[o]="");for(o in l)l.hasOwnProperty(o)&&s[o]!==l[o]&&(n||(n={}),n[o]=l[o])}else n||(a||(a=[]),a.push(u,n)),n=l;else u==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,s=s?s.__html:void 0,l!=null&&s!==l&&(a=a||[]).push(u,l)):u==="children"?typeof l!="string"&&typeof l!="number"||(a=a||[]).push(u,""+l):u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&(zi.hasOwnProperty(u)?(l!=null&&u==="onScroll"&&fe("scroll",e),a||s===l||(a=[])):(a=a||[]).push(u,l))}n&&(a=a||[]).push("style",n);var u=a;(t.updateQueue=u)&&(t.flags|=4)}};nh=function(e,t,n,r){n!==r&&(t.flags|=4)};function ni(e,t){if(!he)switch(e.tailMode){case"hidden":t=e.tail;for(var n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null;break;case"collapsed":n=e.tail;for(var r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null}}function Oe(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&14680064,r|=i.flags&14680064,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function wx(e,t,n){var r=t.pendingProps;switch(Iu(t),t.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return Oe(t),null;case 1:return Ze(t.type)&&oo(),Oe(t),null;case 3:return r=t.stateNode,Ir(),me(Ge),me(qe),Uu(),r.pendingContext&&(r.context=r.pendingContext,r.pendingContext=null),(e===null||e.child===null)&&(_a(t)?t.flags|=4:e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,Nt!==null&&(Hl(Nt),Nt=null))),Dl(e,t),Oe(t),null;case 5:Vu(t);var i=Vn(Mi.current);if(n=t.type,e!==null&&t.stateNode!=null)th(e,t,n,r,i),e.ref!==t.ref&&(t.flags|=512,t.flags|=2097152);else{if(!r){if(t.stateNode===null)throw Error(A(166));return Oe(t),null}if(e=Vn(Bt.current),_a(t)){r=t.stateNode,n=t.type;var a=t.memoizedProps;switch(r[qt]=t,r[Di]=a,e=(t.mode&1)!==0,n){case"dialog":fe("cancel",r),fe("close",r);break;case"iframe":case"object":case"embed":fe("load",r);break;case"video":case"audio":for(i=0;i<pi.length;i++)fe(pi[i],r);break;case"source":fe("error",r);break;case"img":case"image":case"link":fe("error",r),fe("load",r);break;case"details":fe("toggle",r);break;case"input":Fc(r,a),fe("invalid",r);break;case"select":r._wrapperState={wasMultiple:!!a.multiple},fe("invalid",r);break;case"textarea":qc(r,a),fe("invalid",r)}cl(n,a),i=null;for(var o in a)if(a.hasOwnProperty(o)){var s=a[o];o==="children"?typeof s=="string"?r.textContent!==s&&(a.suppressHydrationWarning!==!0&&xa(r.textContent,s,e),i=["children",s]):typeof s=="number"&&r.textContent!==""+s&&(a.suppressHydrationWarning!==!0&&xa(r.textContent,s,e),i=["children",""+s]):zi.hasOwnProperty(o)&&s!=null&&o==="onScroll"&&fe("scroll",r)}switch(n){case"input":ca(r),Mc(r,a,!0);break;case"textarea":ca(r),Wc(r);break;case"select":case"option":break;default:typeof a.onClick=="function"&&(r.onclick=ao)}r=i,t.updateQueue=r,r!==null&&(t.flags|=4)}else{o=i.nodeType===9?i:i.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=Nf(n)),e==="http://www.w3.org/1999/xhtml"?n==="script"?(e=o.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof r.is=="string"?e=o.createElement(n,{is:r.is}):(e=o.createElement(n),n==="select"&&(o=e,r.multiple?o.multiple=!0:r.size&&(o.size=r.size))):e=o.createElementNS(e,n),e[qt]=t,e[Di]=r,eh(e,t,!1,!1),t.stateNode=e;e:{switch(o=dl(n,r),n){case"dialog":fe("cancel",e),fe("close",e),i=r;break;case"iframe":case"object":case"embed":fe("load",e),i=r;break;case"video":case"audio":for(i=0;i<pi.length;i++)fe(pi[i],e);i=r;break;case"source":fe("error",e),i=r;break;case"img":case"image":case"link":fe("error",e),fe("load",e),i=r;break;case"details":fe("toggle",e),i=r;break;case"input":Fc(e,r),i=al(e,r),fe("invalid",e);break;case"option":i=r;break;case"select":e._wrapperState={wasMultiple:!!r.multiple},i=xe({},r,{value:void 0}),fe("invalid",e);break;case"textarea":qc(e,r),i=ll(e,r),fe("invalid",e);break;default:i=r}cl(n,i),s=i;for(a in s)if(s.hasOwnProperty(a)){var l=s[a];a==="style"?Af(e,l):a==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,l!=null&&jf(e,l)):a==="children"?typeof l=="string"?(n!=="textarea"||l!=="")&&Ti(e,l):typeof l=="number"&&Ti(e,""+l):a!=="suppressContentEditableWarning"&&a!=="suppressHydrationWarning"&&a!=="autoFocus"&&(zi.hasOwnProperty(a)?l!=null&&a==="onScroll"&&fe("scroll",e):l!=null&&vu(e,a,l,o))}switch(n){case"input":ca(e),Mc(e,r,!1);break;case"textarea":ca(e),Wc(e);break;case"option":r.value!=null&&e.setAttribute("value",""+Cn(r.value));break;case"select":e.multiple=!!r.multiple,a=r.value,a!=null?br(e,!!r.multiple,a,!1):r.defaultValue!=null&&br(e,!!r.multiple,r.defaultValue,!0);break;default:typeof i.onClick=="function"&&(e.onclick=ao)}switch(n){case"button":case"input":case"select":case"textarea":r=!!r.autoFocus;break e;case"img":r=!0;break e;default:r=!1}}r&&(t.flags|=4)}t.ref!==null&&(t.flags|=512,t.flags|=2097152)}return Oe(t),null;case 6:if(e&&t.stateNode!=null)nh(e,t,e.memoizedProps,r);else{if(typeof r!="string"&&t.stateNode===null)throw Error(A(166));if(n=Vn(Mi.current),Vn(Bt.current),_a(t)){if(r=t.stateNode,n=t.memoizedProps,r[qt]=t,(a=r.nodeValue!==n)&&(e=st,e!==null))switch(e.tag){case 3:xa(r.nodeValue,n,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&xa(r.nodeValue,n,(e.mode&1)!==0)}a&&(t.flags|=4)}else r=(n.nodeType===9?n:n.ownerDocument).createTextNode(r),r[qt]=t,t.stateNode=r}return Oe(t),null;case 13:if(me(ge),r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(he&&at!==null&&t.mode&1&&!(t.flags&128))vm(),Ar(),t.flags|=98560,a=!1;else if(a=_a(t),r!==null&&r.dehydrated!==null){if(e===null){if(!a)throw Error(A(318));if(a=t.memoizedState,a=a!==null?a.dehydrated:null,!a)throw Error(A(317));a[qt]=t}else Ar(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;Oe(t),a=!1}else Nt!==null&&(Hl(Nt),Nt=null),a=!0;if(!a)return t.flags&65536?t:null}return t.flags&128?(t.lanes=n,t):(r=r!==null,r!==(e!==null&&e.memoizedState!==null)&&r&&(t.child.flags|=8192,t.mode&1&&(e===null||ge.current&1?Te===0&&(Te=3):nc())),t.updateQueue!==null&&(t.flags|=4),Oe(t),null);case 4:return Ir(),Dl(e,t),e===null&&Ii(t.stateNode.containerInfo),Oe(t),null;case 10:return Mu(t.type._context),Oe(t),null;case 17:return Ze(t.type)&&oo(),Oe(t),null;case 19:if(me(ge),a=t.memoizedState,a===null)return Oe(t),null;if(r=(t.flags&128)!==0,o=a.rendering,o===null)if(r)ni(a,!1);else{if(Te!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(o=mo(e),o!==null){for(t.flags|=128,ni(a,!1),r=o.updateQueue,r!==null&&(t.updateQueue=r,t.flags|=4),t.subtreeFlags=0,r=n,n=t.child;n!==null;)a=n,e=r,a.flags&=14680066,o=a.alternate,o===null?(a.childLanes=0,a.lanes=e,a.child=null,a.subtreeFlags=0,a.memoizedProps=null,a.memoizedState=null,a.updateQueue=null,a.dependencies=null,a.stateNode=null):(a.childLanes=o.childLanes,a.lanes=o.lanes,a.child=o.child,a.subtreeFlags=0,a.deletions=null,a.memoizedProps=o.memoizedProps,a.memoizedState=o.memoizedState,a.updateQueue=o.updateQueue,a.type=o.type,e=o.dependencies,a.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),n=n.sibling;return pe(ge,ge.current&1|2),t.child}e=e.sibling}a.tail!==null&&ke()>Dr&&(t.flags|=128,r=!0,ni(a,!1),t.lanes=4194304)}else{if(!r)if(e=mo(o),e!==null){if(t.flags|=128,r=!0,n=e.updateQueue,n!==null&&(t.updateQueue=n,t.flags|=4),ni(a,!0),a.tail===null&&a.tailMode==="hidden"&&!o.alternate&&!he)return Oe(t),null}else 2*ke()-a.renderingStartTime>Dr&&n!==1073741824&&(t.flags|=128,r=!0,ni(a,!1),t.lanes=4194304);a.isBackwards?(o.sibling=t.child,t.child=o):(n=a.last,n!==null?n.sibling=o:t.child=o,a.last=o)}return a.tail!==null?(t=a.tail,a.rendering=t,a.tail=t.sibling,a.renderingStartTime=ke(),t.sibling=null,n=ge.current,pe(ge,r?n&1|2:n&1),t):(Oe(t),null);case 22:case 23:return tc(),r=t.memoizedState!==null,e!==null&&e.memoizedState!==null!==r&&(t.flags|=8192),r&&t.mode&1?rt&1073741824&&(Oe(t),t.subtreeFlags&6&&(t.flags|=8192)):Oe(t),null;case 24:return null;case 25:return null}throw Error(A(156,t.tag))}function kx(e,t){switch(Iu(t),t.tag){case 1:return Ze(t.type)&&oo(),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return Ir(),me(Ge),me(qe),Uu(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 5:return Vu(t),null;case 13:if(me(ge),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(A(340));Ar()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return me(ge),null;case 4:return Ir(),null;case 10:return Mu(t.type._context),null;case 22:case 23:return tc(),null;case 24:return null;default:return null}}var wa=!1,Fe=!1,Sx=typeof WeakSet=="function"?WeakSet:Set,V=null;function xr(e,t){var n=e.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(r){ve(e,t,r)}else n.current=null}function Fl(e,t,n){try{n()}catch(r){ve(e,t,r)}}var Nd=!1;function Cx(e,t){if(bl=no,e=sm(),Au(e)){if("selectionStart"in e)var n={start:e.selectionStart,end:e.selectionEnd};else e:{n=(n=e.ownerDocument)&&n.defaultView||window;var r=n.getSelection&&n.getSelection();if(r&&r.rangeCount!==0){n=r.anchorNode;var i=r.anchorOffset,a=r.focusNode;r=r.focusOffset;try{n.nodeType,a.nodeType}catch{n=null;break e}var o=0,s=-1,l=-1,u=0,d=0,c=e,f=null;t:for(;;){for(var m;c!==n||i!==0&&c.nodeType!==3||(s=o+i),c!==a||r!==0&&c.nodeType!==3||(l=o+r),c.nodeType===3&&(o+=c.nodeValue.length),(m=c.firstChild)!==null;)f=c,c=m;for(;;){if(c===e)break t;if(f===n&&++u===i&&(s=o),f===a&&++d===r&&(l=o),(m=c.nextSibling)!==null)break;c=f,f=c.parentNode}c=m}n=s===-1||l===-1?null:{start:s,end:l}}else n=null}n=n||{start:0,end:0}}else n=null;for(wl={focusedElem:e,selectionRange:n},no=!1,V=t;V!==null;)if(t=V,e=t.child,(t.subtreeFlags&1028)!==0&&e!==null)e.return=t,V=e;else for(;V!==null;){t=V;try{var y=t.alternate;if(t.flags&1024)switch(t.tag){case 0:case 11:case 15:break;case 1:if(y!==null){var _=y.memoizedProps,k=y.memoizedState,h=t.stateNode,g=h.getSnapshotBeforeUpdate(t.elementType===t.type?_:Tt(t.type,_),k);h.__reactInternalSnapshotBeforeUpdate=g}break;case 3:var x=t.stateNode.containerInfo;x.nodeType===1?x.textContent="":x.nodeType===9&&x.documentElement&&x.removeChild(x.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(A(163))}}catch(C){ve(t,t.return,C)}if(e=t.sibling,e!==null){e.return=t.return,V=e;break}V=t.return}return y=Nd,Nd=!1,y}function vi(e,t,n){var r=t.updateQueue;if(r=r!==null?r.lastEffect:null,r!==null){var i=r=r.next;do{if((i.tag&e)===e){var a=i.destroy;i.destroy=void 0,a!==void 0&&Fl(t,n,a)}i=i.next}while(i!==r)}}function Fo(e,t){if(t=t.updateQueue,t=t!==null?t.lastEffect:null,t!==null){var n=t=t.next;do{if((n.tag&e)===e){var r=n.create;n.destroy=r()}n=n.next}while(n!==t)}}function Ml(e){var t=e.ref;if(t!==null){var n=e.stateNode;switch(e.tag){case 5:e=n;break;default:e=n}typeof t=="function"?t(e):t.current=e}}function rh(e){var t=e.alternate;t!==null&&(e.alternate=null,rh(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&(delete t[qt],delete t[Di],delete t[Cl],delete t[sx],delete t[lx])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function ih(e){return e.tag===5||e.tag===3||e.tag===4}function jd(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||ih(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function ql(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.nodeType===8?n.parentNode.insertBefore(e,t):n.insertBefore(e,t):(n.nodeType===8?(t=n.parentNode,t.insertBefore(e,n)):(t=n,t.appendChild(e)),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=ao));else if(r!==4&&(e=e.child,e!==null))for(ql(e,t,n),e=e.sibling;e!==null;)ql(e,t,n),e=e.sibling}function Wl(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.insertBefore(e,t):n.appendChild(e);else if(r!==4&&(e=e.child,e!==null))for(Wl(e,t,n),e=e.sibling;e!==null;)Wl(e,t,n),e=e.sibling}var Le=null,Pt=!1;function on(e,t,n){for(n=n.child;n!==null;)ah(e,t,n),n=n.sibling}function ah(e,t,n){if(Wt&&typeof Wt.onCommitFiberUnmount=="function")try{Wt.onCommitFiberUnmount(No,n)}catch{}switch(n.tag){case 5:Fe||xr(n,t);case 6:var r=Le,i=Pt;Le=null,on(e,t,n),Le=r,Pt=i,Le!==null&&(Pt?(e=Le,n=n.stateNode,e.nodeType===8?e.parentNode.removeChild(n):e.removeChild(n)):Le.removeChild(n.stateNode));break;case 18:Le!==null&&(Pt?(e=Le,n=n.stateNode,e.nodeType===8?ys(e.parentNode,n):e.nodeType===1&&ys(e,n),Li(e)):ys(Le,n.stateNode));break;case 4:r=Le,i=Pt,Le=n.stateNode.containerInfo,Pt=!0,on(e,t,n),Le=r,Pt=i;break;case 0:case 11:case 14:case 15:if(!Fe&&(r=n.updateQueue,r!==null&&(r=r.lastEffect,r!==null))){i=r=r.next;do{var a=i,o=a.destroy;a=a.tag,o!==void 0&&(a&2||a&4)&&Fl(n,t,o),i=i.next}while(i!==r)}on(e,t,n);break;case 1:if(!Fe&&(xr(n,t),r=n.stateNode,typeof r.componentWillUnmount=="function"))try{r.props=n.memoizedProps,r.state=n.memoizedState,r.componentWillUnmount()}catch(s){ve(n,t,s)}on(e,t,n);break;case 21:on(e,t,n);break;case 22:n.mode&1?(Fe=(r=Fe)||n.memoizedState!==null,on(e,t,n),Fe=r):on(e,t,n);break;default:on(e,t,n)}}function Ld(e){var t=e.updateQueue;if(t!==null){e.updateQueue=null;var n=e.stateNode;n===null&&(n=e.stateNode=new Sx),t.forEach(function(r){var i=Rx.bind(null,e,r);n.has(r)||(n.add(r),r.then(i,i))})}}function kt(e,t){var n=t.deletions;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];try{var a=e,o=t,s=o;e:for(;s!==null;){switch(s.tag){case 5:Le=s.stateNode,Pt=!1;break e;case 3:Le=s.stateNode.containerInfo,Pt=!0;break e;case 4:Le=s.stateNode.containerInfo,Pt=!0;break e}s=s.return}if(Le===null)throw Error(A(160));ah(a,o,i),Le=null,Pt=!1;var l=i.alternate;l!==null&&(l.return=null),i.return=null}catch(u){ve(i,t,u)}}if(t.subtreeFlags&12854)for(t=t.child;t!==null;)oh(t,e),t=t.sibling}function oh(e,t){var n=e.alternate,r=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(kt(t,e),Dt(e),r&4){try{vi(3,e,e.return),Fo(3,e)}catch(_){ve(e,e.return,_)}try{vi(5,e,e.return)}catch(_){ve(e,e.return,_)}}break;case 1:kt(t,e),Dt(e),r&512&&n!==null&&xr(n,n.return);break;case 5:if(kt(t,e),Dt(e),r&512&&n!==null&&xr(n,n.return),e.flags&32){var i=e.stateNode;try{Ti(i,"")}catch(_){ve(e,e.return,_)}}if(r&4&&(i=e.stateNode,i!=null)){var a=e.memoizedProps,o=n!==null?n.memoizedProps:a,s=e.type,l=e.updateQueue;if(e.updateQueue=null,l!==null)try{s==="input"&&a.type==="radio"&&a.name!=null&&Tf(i,a),dl(s,o);var u=dl(s,a);for(o=0;o<l.length;o+=2){var d=l[o],c=l[o+1];d==="style"?Af(i,c):d==="dangerouslySetInnerHTML"?jf(i,c):d==="children"?Ti(i,c):vu(i,d,c,u)}switch(s){case"input":ol(i,a);break;case"textarea":Pf(i,a);break;case"select":var f=i._wrapperState.wasMultiple;i._wrapperState.wasMultiple=!!a.multiple;var m=a.value;m!=null?br(i,!!a.multiple,m,!1):f!==!!a.multiple&&(a.defaultValue!=null?br(i,!!a.multiple,a.defaultValue,!0):br(i,!!a.multiple,a.multiple?[]:"",!1))}i[Di]=a}catch(_){ve(e,e.return,_)}}break;case 6:if(kt(t,e),Dt(e),r&4){if(e.stateNode===null)throw Error(A(162));i=e.stateNode,a=e.memoizedProps;try{i.nodeValue=a}catch(_){ve(e,e.return,_)}}break;case 3:if(kt(t,e),Dt(e),r&4&&n!==null&&n.memoizedState.isDehydrated)try{Li(t.containerInfo)}catch(_){ve(e,e.return,_)}break;case 4:kt(t,e),Dt(e);break;case 13:kt(t,e),Dt(e),i=e.child,i.flags&8192&&(a=i.memoizedState!==null,i.stateNode.isHidden=a,!a||i.alternate!==null&&i.alternate.memoizedState!==null||(Ju=ke())),r&4&&Ld(e);break;case 22:if(d=n!==null&&n.memoizedState!==null,e.mode&1?(Fe=(u=Fe)||d,kt(t,e),Fe=u):kt(t,e),Dt(e),r&8192){if(u=e.memoizedState!==null,(e.stateNode.isHidden=u)&&!d&&e.mode&1)for(V=e,d=e.child;d!==null;){for(c=V=d;V!==null;){switch(f=V,m=f.child,f.tag){case 0:case 11:case 14:case 15:vi(4,f,f.return);break;case 1:xr(f,f.return);var y=f.stateNode;if(typeof y.componentWillUnmount=="function"){r=f,n=f.return;try{t=r,y.props=t.memoizedProps,y.state=t.memoizedState,y.componentWillUnmount()}catch(_){ve(r,n,_)}}break;case 5:xr(f,f.return);break;case 22:if(f.memoizedState!==null){Rd(c);continue}}m!==null?(m.return=f,V=m):Rd(c)}d=d.sibling}e:for(d=null,c=e;;){if(c.tag===5){if(d===null){d=c;try{i=c.stateNode,u?(a=i.style,typeof a.setProperty=="function"?a.setProperty("display","none","important"):a.display="none"):(s=c.stateNode,l=c.memoizedProps.style,o=l!=null&&l.hasOwnProperty("display")?l.display:null,s.style.display=Lf("display",o))}catch(_){ve(e,e.return,_)}}}else if(c.tag===6){if(d===null)try{c.stateNode.nodeValue=u?"":c.memoizedProps}catch(_){ve(e,e.return,_)}}else if((c.tag!==22&&c.tag!==23||c.memoizedState===null||c===e)&&c.child!==null){c.child.return=c,c=c.child;continue}if(c===e)break e;for(;c.sibling===null;){if(c.return===null||c.return===e)break e;d===c&&(d=null),c=c.return}d===c&&(d=null),c.sibling.return=c.return,c=c.sibling}}break;case 19:kt(t,e),Dt(e),r&4&&Ld(e);break;case 21:break;default:kt(t,e),Dt(e)}}function Dt(e){var t=e.flags;if(t&2){try{e:{for(var n=e.return;n!==null;){if(ih(n)){var r=n;break e}n=n.return}throw Error(A(160))}switch(r.tag){case 5:var i=r.stateNode;r.flags&32&&(Ti(i,""),r.flags&=-33);var a=jd(e);Wl(e,a,i);break;case 3:case 4:var o=r.stateNode.containerInfo,s=jd(e);ql(e,s,o);break;default:throw Error(A(161))}}catch(l){ve(e,e.return,l)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function Ex(e,t,n){V=e,sh(e)}function sh(e,t,n){for(var r=(e.mode&1)!==0;V!==null;){var i=V,a=i.child;if(i.tag===22&&r){var o=i.memoizedState!==null||wa;if(!o){var s=i.alternate,l=s!==null&&s.memoizedState!==null||Fe;s=wa;var u=Fe;if(wa=o,(Fe=l)&&!u)for(V=i;V!==null;)o=V,l=o.child,o.tag===22&&o.memoizedState!==null?Id(i):l!==null?(l.return=o,V=l):Id(i);for(;a!==null;)V=a,sh(a),a=a.sibling;V=i,wa=s,Fe=u}Ad(e)}else i.subtreeFlags&8772&&a!==null?(a.return=i,V=a):Ad(e)}}function Ad(e){for(;V!==null;){var t=V;if(t.flags&8772){var n=t.alternate;try{if(t.flags&8772)switch(t.tag){case 0:case 11:case 15:Fe||Fo(5,t);break;case 1:var r=t.stateNode;if(t.flags&4&&!Fe)if(n===null)r.componentDidMount();else{var i=t.elementType===t.type?n.memoizedProps:Tt(t.type,n.memoizedProps);r.componentDidUpdate(i,n.memoizedState,r.__reactInternalSnapshotBeforeUpdate)}var a=t.updateQueue;a!==null&&yd(t,a,r);break;case 3:var o=t.updateQueue;if(o!==null){if(n=null,t.child!==null)switch(t.child.tag){case 5:n=t.child.stateNode;break;case 1:n=t.child.stateNode}yd(t,o,n)}break;case 5:var s=t.stateNode;if(n===null&&t.flags&4){n=s;var l=t.memoizedProps;switch(t.type){case"button":case"input":case"select":case"textarea":l.autoFocus&&n.focus();break;case"img":l.src&&(n.src=l.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(t.memoizedState===null){var u=t.alternate;if(u!==null){var d=u.memoizedState;if(d!==null){var c=d.dehydrated;c!==null&&Li(c)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(A(163))}Fe||t.flags&512&&Ml(t)}catch(f){ve(t,t.return,f)}}if(t===e){V=null;break}if(n=t.sibling,n!==null){n.return=t.return,V=n;break}V=t.return}}function Rd(e){for(;V!==null;){var t=V;if(t===e){V=null;break}var n=t.sibling;if(n!==null){n.return=t.return,V=n;break}V=t.return}}function Id(e){for(;V!==null;){var t=V;try{switch(t.tag){case 0:case 11:case 15:var n=t.return;try{Fo(4,t)}catch(l){ve(t,n,l)}break;case 1:var r=t.stateNode;if(typeof r.componentDidMount=="function"){var i=t.return;try{r.componentDidMount()}catch(l){ve(t,i,l)}}var a=t.return;try{Ml(t)}catch(l){ve(t,a,l)}break;case 5:var o=t.return;try{Ml(t)}catch(l){ve(t,o,l)}}}catch(l){ve(t,t.return,l)}if(t===e){V=null;break}var s=t.sibling;if(s!==null){s.return=t.return,V=s;break}V=t.return}}var zx=Math.ceil,yo=rn.ReactCurrentDispatcher,Gu=rn.ReactCurrentOwner,_t=rn.ReactCurrentBatchConfig,ae=0,je=null,Ce=null,Ae=0,rt=0,_r=Tn(0),Te=0,Vi=null,Xn=0,Mo=0,Zu=0,bi=null,Ye=null,Ju=0,Dr=1/0,Kt=null,xo=!1,Bl=null,vn=null,ka=!1,fn=null,_o=0,wi=0,Vl=null,Va=-1,Ua=0;function He(){return ae&6?ke():Va!==-1?Va:Va=ke()}function bn(e){return e.mode&1?ae&2&&Ae!==0?Ae&-Ae:cx.transition!==null?(Ua===0&&(Ua=Hf()),Ua):(e=se,e!==0||(e=window.event,e=e===void 0?16:Zf(e.type)),e):1}function At(e,t,n,r){if(50<wi)throw wi=0,Vl=null,Error(A(185));Gi(e,n,r),(!(ae&2)||e!==je)&&(e===je&&(!(ae&2)&&(Mo|=n),Te===4&&dn(e,Ae)),Je(e,r),n===1&&ae===0&&!(t.mode&1)&&(Dr=ke()+500,Io&&Pn()))}function Je(e,t){var n=e.callbackNode;c0(e,t);var r=to(e,e===je?Ae:0);if(r===0)n!==null&&Uc(n),e.callbackNode=null,e.callbackPriority=0;else if(t=r&-r,e.callbackPriority!==t){if(n!=null&&Uc(n),t===1)e.tag===0?ux(Od.bind(null,e)):ym(Od.bind(null,e)),ax(function(){!(ae&6)&&Pn()}),n=null;else{switch($f(r)){case 1:n=Cu;break;case 4:n=Vf;break;case 16:n=eo;break;case 536870912:n=Uf;break;default:n=eo}n=hh(n,lh.bind(null,e))}e.callbackPriority=t,e.callbackNode=n}}function lh(e,t){if(Va=-1,Ua=0,ae&6)throw Error(A(327));var n=e.callbackNode;if(Er()&&e.callbackNode!==n)return null;var r=to(e,e===je?Ae:0);if(r===0)return null;if(r&30||r&e.expiredLanes||t)t=vo(e,r);else{t=r;var i=ae;ae|=2;var a=ch();(je!==e||Ae!==t)&&(Kt=null,Dr=ke()+500,Hn(e,t));do try{Nx();break}catch(s){uh(e,s)}while(!0);Fu(),yo.current=a,ae=i,Ce!==null?t=0:(je=null,Ae=0,t=Te)}if(t!==0){if(t===2&&(i=gl(e),i!==0&&(r=i,t=Ul(e,i))),t===1)throw n=Vi,Hn(e,0),dn(e,r),Je(e,ke()),n;if(t===6)dn(e,r);else{if(i=e.current.alternate,!(r&30)&&!Tx(i)&&(t=vo(e,r),t===2&&(a=gl(e),a!==0&&(r=a,t=Ul(e,a))),t===1))throw n=Vi,Hn(e,0),dn(e,r),Je(e,ke()),n;switch(e.finishedWork=i,e.finishedLanes=r,t){case 0:case 1:throw Error(A(345));case 2:Fn(e,Ye,Kt);break;case 3:if(dn(e,r),(r&130023424)===r&&(t=Ju+500-ke(),10<t)){if(to(e,0)!==0)break;if(i=e.suspendedLanes,(i&r)!==r){He(),e.pingedLanes|=e.suspendedLanes&i;break}e.timeoutHandle=Sl(Fn.bind(null,e,Ye,Kt),t);break}Fn(e,Ye,Kt);break;case 4:if(dn(e,r),(r&4194240)===r)break;for(t=e.eventTimes,i=-1;0<r;){var o=31-Lt(r);a=1<<o,o=t[o],o>i&&(i=o),r&=~a}if(r=i,r=ke()-r,r=(120>r?120:480>r?480:1080>r?1080:1920>r?1920:3e3>r?3e3:4320>r?4320:1960*zx(r/1960))-r,10<r){e.timeoutHandle=Sl(Fn.bind(null,e,Ye,Kt),r);break}Fn(e,Ye,Kt);break;case 5:Fn(e,Ye,Kt);break;default:throw Error(A(329))}}}return Je(e,ke()),e.callbackNode===n?lh.bind(null,e):null}function Ul(e,t){var n=bi;return e.current.memoizedState.isDehydrated&&(Hn(e,t).flags|=256),e=vo(e,t),e!==2&&(t=Ye,Ye=n,t!==null&&Hl(t)),e}function Hl(e){Ye===null?Ye=e:Ye.push.apply(Ye,e)}function Tx(e){for(var t=e;;){if(t.flags&16384){var n=t.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var r=0;r<n.length;r++){var i=n[r],a=i.getSnapshot;i=i.value;try{if(!It(a(),i))return!1}catch{return!1}}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function dn(e,t){for(t&=~Zu,t&=~Mo,e.suspendedLanes|=t,e.pingedLanes&=~t,e=e.expirationTimes;0<t;){var n=31-Lt(t),r=1<<n;e[n]=-1,t&=~r}}function Od(e){if(ae&6)throw Error(A(327));Er();var t=to(e,0);if(!(t&1))return Je(e,ke()),null;var n=vo(e,t);if(e.tag!==0&&n===2){var r=gl(e);r!==0&&(t=r,n=Ul(e,r))}if(n===1)throw n=Vi,Hn(e,0),dn(e,t),Je(e,ke()),n;if(n===6)throw Error(A(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,Fn(e,Ye,Kt),Je(e,ke()),null}function ec(e,t){var n=ae;ae|=1;try{return e(t)}finally{ae=n,ae===0&&(Dr=ke()+500,Io&&Pn())}}function Gn(e){fn!==null&&fn.tag===0&&!(ae&6)&&Er();var t=ae;ae|=1;var n=_t.transition,r=se;try{if(_t.transition=null,se=1,e)return e()}finally{se=r,_t.transition=n,ae=t,!(ae&6)&&Pn()}}function tc(){rt=_r.current,me(_r)}function Hn(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(n!==-1&&(e.timeoutHandle=-1,ix(n)),Ce!==null)for(n=Ce.return;n!==null;){var r=n;switch(Iu(r),r.tag){case 1:r=r.type.childContextTypes,r!=null&&oo();break;case 3:Ir(),me(Ge),me(qe),Uu();break;case 5:Vu(r);break;case 4:Ir();break;case 13:me(ge);break;case 19:me(ge);break;case 10:Mu(r.type._context);break;case 22:case 23:tc()}n=n.return}if(je=e,Ce=e=wn(e.current,null),Ae=rt=t,Te=0,Vi=null,Zu=Mo=Xn=0,Ye=bi=null,Bn!==null){for(t=0;t<Bn.length;t++)if(n=Bn[t],r=n.interleaved,r!==null){n.interleaved=null;var i=r.next,a=n.pending;if(a!==null){var o=a.next;a.next=i,r.next=o}n.pending=r}Bn=null}return e}function uh(e,t){do{var n=Ce;try{if(Fu(),qa.current=go,ho){for(var r=ye.memoizedState;r!==null;){var i=r.queue;i!==null&&(i.pending=null),r=r.next}ho=!1}if(Yn=0,Pe=ze=ye=null,_i=!1,qi=0,Gu.current=null,n===null||n.return===null){Te=1,Vi=t,Ce=null;break}e:{var a=e,o=n.return,s=n,l=t;if(t=Ae,s.flags|=32768,l!==null&&typeof l=="object"&&typeof l.then=="function"){var u=l,d=s,c=d.tag;if(!(d.mode&1)&&(c===0||c===11||c===15)){var f=d.alternate;f?(d.updateQueue=f.updateQueue,d.memoizedState=f.memoizedState,d.lanes=f.lanes):(d.updateQueue=null,d.memoizedState=null)}var m=kd(o);if(m!==null){m.flags&=-257,Sd(m,o,s,a,t),m.mode&1&&wd(a,u,t),t=m,l=u;var y=t.updateQueue;if(y===null){var _=new Set;_.add(l),t.updateQueue=_}else y.add(l);break e}else{if(!(t&1)){wd(a,u,t),nc();break e}l=Error(A(426))}}else if(he&&s.mode&1){var k=kd(o);if(k!==null){!(k.flags&65536)&&(k.flags|=256),Sd(k,o,s,a,t),Ou(Or(l,s));break e}}a=l=Or(l,s),Te!==4&&(Te=2),bi===null?bi=[a]:bi.push(a),a=o;do{switch(a.tag){case 3:a.flags|=65536,t&=-t,a.lanes|=t;var h=$m(a,l,t);gd(a,h);break e;case 1:s=l;var g=a.type,x=a.stateNode;if(!(a.flags&128)&&(typeof g.getDerivedStateFromError=="function"||x!==null&&typeof x.componentDidCatch=="function"&&(vn===null||!vn.has(x)))){a.flags|=65536,t&=-t,a.lanes|=t;var C=Qm(a,s,t);gd(a,C);break e}}a=a.return}while(a!==null)}ph(n)}catch(z){t=z,Ce===n&&n!==null&&(Ce=n=n.return);continue}break}while(!0)}function ch(){var e=yo.current;return yo.current=go,e===null?go:e}function nc(){(Te===0||Te===3||Te===2)&&(Te=4),je===null||!(Xn&268435455)&&!(Mo&268435455)||dn(je,Ae)}function vo(e,t){var n=ae;ae|=2;var r=ch();(je!==e||Ae!==t)&&(Kt=null,Hn(e,t));do try{Px();break}catch(i){uh(e,i)}while(!0);if(Fu(),ae=n,yo.current=r,Ce!==null)throw Error(A(261));return je=null,Ae=0,Te}function Px(){for(;Ce!==null;)dh(Ce)}function Nx(){for(;Ce!==null&&!t0();)dh(Ce)}function dh(e){var t=mh(e.alternate,e,rt);e.memoizedProps=e.pendingProps,t===null?ph(e):Ce=t,Gu.current=null}function ph(e){var t=e;do{var n=t.alternate;if(e=t.return,t.flags&32768){if(n=kx(n,t),n!==null){n.flags&=32767,Ce=n;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{Te=6,Ce=null;return}}else if(n=wx(n,t,rt),n!==null){Ce=n;return}if(t=t.sibling,t!==null){Ce=t;return}Ce=t=e}while(t!==null);Te===0&&(Te=5)}function Fn(e,t,n){var r=se,i=_t.transition;try{_t.transition=null,se=1,jx(e,t,n,r)}finally{_t.transition=i,se=r}return null}function jx(e,t,n,r){do Er();while(fn!==null);if(ae&6)throw Error(A(327));n=e.finishedWork;var i=e.finishedLanes;if(n===null)return null;if(e.finishedWork=null,e.finishedLanes=0,n===e.current)throw Error(A(177));e.callbackNode=null,e.callbackPriority=0;var a=n.lanes|n.childLanes;if(d0(e,a),e===je&&(Ce=je=null,Ae=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||ka||(ka=!0,hh(eo,function(){return Er(),null})),a=(n.flags&15990)!==0,n.subtreeFlags&15990||a){a=_t.transition,_t.transition=null;var o=se;se=1;var s=ae;ae|=4,Gu.current=null,Cx(e,n),oh(n,e),G0(wl),no=!!bl,wl=bl=null,e.current=n,Ex(n),n0(),ae=s,se=o,_t.transition=a}else e.current=n;if(ka&&(ka=!1,fn=e,_o=i),a=e.pendingLanes,a===0&&(vn=null),a0(n.stateNode),Je(e,ke()),t!==null)for(r=e.onRecoverableError,n=0;n<t.length;n++)i=t[n],r(i.value,{componentStack:i.stack,digest:i.digest});if(xo)throw xo=!1,e=Bl,Bl=null,e;return _o&1&&e.tag!==0&&Er(),a=e.pendingLanes,a&1?e===Vl?wi++:(wi=0,Vl=e):wi=0,Pn(),null}function Er(){if(fn!==null){var e=$f(_o),t=_t.transition,n=se;try{if(_t.transition=null,se=16>e?16:e,fn===null)var r=!1;else{if(e=fn,fn=null,_o=0,ae&6)throw Error(A(331));var i=ae;for(ae|=4,V=e.current;V!==null;){var a=V,o=a.child;if(V.flags&16){var s=a.deletions;if(s!==null){for(var l=0;l<s.length;l++){var u=s[l];for(V=u;V!==null;){var d=V;switch(d.tag){case 0:case 11:case 15:vi(8,d,a)}var c=d.child;if(c!==null)c.return=d,V=c;else for(;V!==null;){d=V;var f=d.sibling,m=d.return;if(rh(d),d===u){V=null;break}if(f!==null){f.return=m,V=f;break}V=m}}}var y=a.alternate;if(y!==null){var _=y.child;if(_!==null){y.child=null;do{var k=_.sibling;_.sibling=null,_=k}while(_!==null)}}V=a}}if(a.subtreeFlags&2064&&o!==null)o.return=a,V=o;else e:for(;V!==null;){if(a=V,a.flags&2048)switch(a.tag){case 0:case 11:case 15:vi(9,a,a.return)}var h=a.sibling;if(h!==null){h.return=a.return,V=h;break e}V=a.return}}var g=e.current;for(V=g;V!==null;){o=V;var x=o.child;if(o.subtreeFlags&2064&&x!==null)x.return=o,V=x;else e:for(o=g;V!==null;){if(s=V,s.flags&2048)try{switch(s.tag){case 0:case 11:case 15:Fo(9,s)}}catch(z){ve(s,s.return,z)}if(s===o){V=null;break e}var C=s.sibling;if(C!==null){C.return=s.return,V=C;break e}V=s.return}}if(ae=i,Pn(),Wt&&typeof Wt.onPostCommitFiberRoot=="function")try{Wt.onPostCommitFiberRoot(No,e)}catch{}r=!0}return r}finally{se=n,_t.transition=t}}return!1}function Dd(e,t,n){t=Or(n,t),t=$m(e,t,1),e=_n(e,t,1),t=He(),e!==null&&(Gi(e,1,t),Je(e,t))}function ve(e,t,n){if(e.tag===3)Dd(e,e,n);else for(;t!==null;){if(t.tag===3){Dd(t,e,n);break}else if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError=="function"||typeof r.componentDidCatch=="function"&&(vn===null||!vn.has(r))){e=Or(n,e),e=Qm(t,e,1),t=_n(t,e,1),e=He(),t!==null&&(Gi(t,1,e),Je(t,e));break}}t=t.return}}function Lx(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),t=He(),e.pingedLanes|=e.suspendedLanes&n,je===e&&(Ae&n)===n&&(Te===4||Te===3&&(Ae&130023424)===Ae&&500>ke()-Ju?Hn(e,0):Zu|=n),Je(e,t)}function fh(e,t){t===0&&(e.mode&1?(t=fa,fa<<=1,!(fa&130023424)&&(fa=4194304)):t=1);var n=He();e=tn(e,t),e!==null&&(Gi(e,t,n),Je(e,n))}function Ax(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),fh(e,n)}function Rx(e,t){var n=0;switch(e.tag){case 13:var r=e.stateNode,i=e.memoizedState;i!==null&&(n=i.retryLane);break;case 19:r=e.stateNode;break;default:throw Error(A(314))}r!==null&&r.delete(t),fh(e,n)}var mh;mh=function(e,t,n){if(e!==null)if(e.memoizedProps!==t.pendingProps||Ge.current)Xe=!0;else{if(!(e.lanes&n)&&!(t.flags&128))return Xe=!1,bx(e,t,n);Xe=!!(e.flags&131072)}else Xe=!1,he&&t.flags&1048576&&xm(t,uo,t.index);switch(t.lanes=0,t.tag){case 2:var r=t.type;Ba(e,t),e=t.pendingProps;var i=Lr(t,qe.current);Cr(t,n),i=$u(null,t,r,e,i,n);var a=Qu();return t.flags|=1,typeof i=="object"&&i!==null&&typeof i.render=="function"&&i.$$typeof===void 0?(t.tag=1,t.memoizedState=null,t.updateQueue=null,Ze(r)?(a=!0,so(t)):a=!1,t.memoizedState=i.state!==null&&i.state!==void 0?i.state:null,Wu(t),i.updater=Do,t.stateNode=i,i._reactInternals=t,jl(t,r,e,n),t=Rl(null,t,r,!0,a,n)):(t.tag=0,he&&a&&Ru(t),Ve(null,t,i,n),t=t.child),t;case 16:r=t.elementType;e:{switch(Ba(e,t),e=t.pendingProps,i=r._init,r=i(r._payload),t.type=r,i=t.tag=Ox(r),e=Tt(r,e),i){case 0:t=Al(null,t,r,e,n);break e;case 1:t=zd(null,t,r,e,n);break e;case 11:t=Cd(null,t,r,e,n);break e;case 14:t=Ed(null,t,r,Tt(r.type,e),n);break e}throw Error(A(306,r,""))}return t;case 0:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Tt(r,i),Al(e,t,r,i,n);case 1:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Tt(r,i),zd(e,t,r,i,n);case 3:e:{if(Gm(t),e===null)throw Error(A(387));r=t.pendingProps,a=t.memoizedState,i=a.element,Sm(e,t),fo(t,r,null,n);var o=t.memoizedState;if(r=o.element,a.isDehydrated)if(a={element:r,isDehydrated:!1,cache:o.cache,pendingSuspenseBoundaries:o.pendingSuspenseBoundaries,transitions:o.transitions},t.updateQueue.baseState=a,t.memoizedState=a,t.flags&256){i=Or(Error(A(423)),t),t=Td(e,t,r,n,i);break e}else if(r!==i){i=Or(Error(A(424)),t),t=Td(e,t,r,n,i);break e}else for(at=xn(t.stateNode.containerInfo.firstChild),st=t,he=!0,Nt=null,n=wm(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(Ar(),r===i){t=nn(e,t,n);break e}Ve(e,t,r,n)}t=t.child}return t;case 5:return Cm(t),e===null&&Tl(t),r=t.type,i=t.pendingProps,a=e!==null?e.memoizedProps:null,o=i.children,kl(r,i)?o=null:a!==null&&kl(r,a)&&(t.flags|=32),Xm(e,t),Ve(e,t,o,n),t.child;case 6:return e===null&&Tl(t),null;case 13:return Zm(e,t,n);case 4:return Bu(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=Rr(t,null,r,n):Ve(e,t,r,n),t.child;case 11:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Tt(r,i),Cd(e,t,r,i,n);case 7:return Ve(e,t,t.pendingProps,n),t.child;case 8:return Ve(e,t,t.pendingProps.children,n),t.child;case 12:return Ve(e,t,t.pendingProps.children,n),t.child;case 10:e:{if(r=t.type._context,i=t.pendingProps,a=t.memoizedProps,o=i.value,pe(co,r._currentValue),r._currentValue=o,a!==null)if(It(a.value,o)){if(a.children===i.children&&!Ge.current){t=nn(e,t,n);break e}}else for(a=t.child,a!==null&&(a.return=t);a!==null;){var s=a.dependencies;if(s!==null){o=a.child;for(var l=s.firstContext;l!==null;){if(l.context===r){if(a.tag===1){l=Zt(-1,n&-n),l.tag=2;var u=a.updateQueue;if(u!==null){u=u.shared;var d=u.pending;d===null?l.next=l:(l.next=d.next,d.next=l),u.pending=l}}a.lanes|=n,l=a.alternate,l!==null&&(l.lanes|=n),Pl(a.return,n,t),s.lanes|=n;break}l=l.next}}else if(a.tag===10)o=a.type===t.type?null:a.child;else if(a.tag===18){if(o=a.return,o===null)throw Error(A(341));o.lanes|=n,s=o.alternate,s!==null&&(s.lanes|=n),Pl(o,n,t),o=a.sibling}else o=a.child;if(o!==null)o.return=a;else for(o=a;o!==null;){if(o===t){o=null;break}if(a=o.sibling,a!==null){a.return=o.return,o=a;break}o=o.return}a=o}Ve(e,t,i.children,n),t=t.child}return t;case 9:return i=t.type,r=t.pendingProps.children,Cr(t,n),i=vt(i),r=r(i),t.flags|=1,Ve(e,t,r,n),t.child;case 14:return r=t.type,i=Tt(r,t.pendingProps),i=Tt(r.type,i),Ed(e,t,r,i,n);case 15:return Km(e,t,t.type,t.pendingProps,n);case 17:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Tt(r,i),Ba(e,t),t.tag=1,Ze(r)?(e=!0,so(t)):e=!1,Cr(t,n),Hm(t,r,i),jl(t,r,i,n),Rl(null,t,r,!0,e,n);case 19:return Jm(e,t,n);case 22:return Ym(e,t,n)}throw Error(A(156,t.tag))};function hh(e,t){return Bf(e,t)}function Ix(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function xt(e,t,n,r){return new Ix(e,t,n,r)}function rc(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Ox(e){if(typeof e=="function")return rc(e)?1:0;if(e!=null){if(e=e.$$typeof,e===wu)return 11;if(e===ku)return 14}return 2}function wn(e,t){var n=e.alternate;return n===null?(n=xt(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&14680064,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}function Ha(e,t,n,r,i,a){var o=2;if(r=e,typeof e=="function")rc(e)&&(o=1);else if(typeof e=="string")o=5;else e:switch(e){case ur:return $n(n.children,i,a,t);case bu:o=8,i|=8;break;case tl:return e=xt(12,n,t,i|2),e.elementType=tl,e.lanes=a,e;case nl:return e=xt(13,n,t,i),e.elementType=nl,e.lanes=a,e;case rl:return e=xt(19,n,t,i),e.elementType=rl,e.lanes=a,e;case Cf:return qo(n,i,a,t);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case kf:o=10;break e;case Sf:o=9;break e;case wu:o=11;break e;case ku:o=14;break e;case ln:o=16,r=null;break e}throw Error(A(130,e==null?e:typeof e,""))}return t=xt(o,n,t,i),t.elementType=e,t.type=r,t.lanes=a,t}function $n(e,t,n,r){return e=xt(7,e,r,t),e.lanes=n,e}function qo(e,t,n,r){return e=xt(22,e,r,t),e.elementType=Cf,e.lanes=n,e.stateNode={isHidden:!1},e}function Cs(e,t,n){return e=xt(6,e,null,t),e.lanes=n,e}function Es(e,t,n){return t=xt(4,e.children!==null?e.children:[],e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}function Dx(e,t,n,r,i){this.tag=t,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=os(0),this.expirationTimes=os(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=os(0),this.identifierPrefix=r,this.onRecoverableError=i,this.mutableSourceEagerHydrationData=null}function ic(e,t,n,r,i,a,o,s,l){return e=new Dx(e,t,n,s,l),t===1?(t=1,a===!0&&(t|=8)):t=0,a=xt(3,null,null,t),e.current=a,a.stateNode=e,a.memoizedState={element:r,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},Wu(a),e}function Fx(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:lr,key:r==null?null:""+r,children:e,containerInfo:t,implementation:n}}function gh(e){if(!e)return En;e=e._reactInternals;e:{if(tr(e)!==e||e.tag!==1)throw Error(A(170));var t=e;do{switch(t.tag){case 3:t=t.stateNode.context;break e;case 1:if(Ze(t.type)){t=t.stateNode.__reactInternalMemoizedMergedChildContext;break e}}t=t.return}while(t!==null);throw Error(A(171))}if(e.tag===1){var n=e.type;if(Ze(n))return gm(e,n,t)}return t}function yh(e,t,n,r,i,a,o,s,l){return e=ic(n,r,!0,e,i,a,o,s,l),e.context=gh(null),n=e.current,r=He(),i=bn(n),a=Zt(r,i),a.callback=t??null,_n(n,a,i),e.current.lanes=i,Gi(e,i,r),Je(e,r),e}function Wo(e,t,n,r){var i=t.current,a=He(),o=bn(i);return n=gh(n),t.context===null?t.context=n:t.pendingContext=n,t=Zt(a,o),t.payload={element:e},r=r===void 0?null:r,r!==null&&(t.callback=r),e=_n(i,t,o),e!==null&&(At(e,i,o,a),Ma(e,i,o)),o}function bo(e){if(e=e.current,!e.child)return null;switch(e.child.tag){case 5:return e.child.stateNode;default:return e.child.stateNode}}function Fd(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function ac(e,t){Fd(e,t),(e=e.alternate)&&Fd(e,t)}function Mx(){return null}var xh=typeof reportError=="function"?reportError:function(e){console.error(e)};function oc(e){this._internalRoot=e}Bo.prototype.render=oc.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(A(409));Wo(e,t,null,null)};Bo.prototype.unmount=oc.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;Gn(function(){Wo(null,e,null,null)}),t[en]=null}};function Bo(e){this._internalRoot=e}Bo.prototype.unstable_scheduleHydration=function(e){if(e){var t=Yf();e={blockedOn:null,target:e,priority:t};for(var n=0;n<cn.length&&t!==0&&t<cn[n].priority;n++);cn.splice(n,0,e),n===0&&Gf(e)}};function sc(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function Vo(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function Md(){}function qx(e,t,n,r,i){if(i){if(typeof r=="function"){var a=r;r=function(){var u=bo(o);a.call(u)}}var o=yh(t,r,e,0,null,!1,!1,"",Md);return e._reactRootContainer=o,e[en]=o.current,Ii(e.nodeType===8?e.parentNode:e),Gn(),o}for(;i=e.lastChild;)e.removeChild(i);if(typeof r=="function"){var s=r;r=function(){var u=bo(l);s.call(u)}}var l=ic(e,0,!1,null,null,!1,!1,"",Md);return e._reactRootContainer=l,e[en]=l.current,Ii(e.nodeType===8?e.parentNode:e),Gn(function(){Wo(t,l,n,r)}),l}function Uo(e,t,n,r,i){var a=n._reactRootContainer;if(a){var o=a;if(typeof i=="function"){var s=i;i=function(){var l=bo(o);s.call(l)}}Wo(t,o,e,i)}else o=qx(n,t,e,i,r);return bo(o)}Qf=function(e){switch(e.tag){case 3:var t=e.stateNode;if(t.current.memoizedState.isDehydrated){var n=di(t.pendingLanes);n!==0&&(Eu(t,n|1),Je(t,ke()),!(ae&6)&&(Dr=ke()+500,Pn()))}break;case 13:Gn(function(){var r=tn(e,1);if(r!==null){var i=He();At(r,e,1,i)}}),ac(e,1)}};zu=function(e){if(e.tag===13){var t=tn(e,134217728);if(t!==null){var n=He();At(t,e,134217728,n)}ac(e,134217728)}};Kf=function(e){if(e.tag===13){var t=bn(e),n=tn(e,t);if(n!==null){var r=He();At(n,e,t,r)}ac(e,t)}};Yf=function(){return se};Xf=function(e,t){var n=se;try{return se=e,t()}finally{se=n}};fl=function(e,t,n){switch(t){case"input":if(ol(e,n),t=n.name,n.type==="radio"&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+t)+'][type="radio"]'),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var i=Ro(r);if(!i)throw Error(A(90));zf(r),ol(r,i)}}}break;case"textarea":Pf(e,n);break;case"select":t=n.value,t!=null&&br(e,!!n.multiple,t,!1)}};Of=ec;Df=Gn;var Wx={usingClientEntryPoint:!1,Events:[Ji,fr,Ro,Rf,If,ec]},ri={findFiberByHostInstance:Wn,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},Bx={bundleType:ri.bundleType,version:ri.version,rendererPackageName:ri.rendererPackageName,rendererConfig:ri.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:rn.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=qf(e),e===null?null:e.stateNode},findFiberByHostInstance:ri.findFiberByHostInstance||Mx,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var Sa=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Sa.isDisabled&&Sa.supportsFiber)try{No=Sa.inject(Bx),Wt=Sa}catch{}}ct.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=Wx;ct.createPortal=function(e,t){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!sc(t))throw Error(A(200));return Fx(e,t,null,n)};ct.createRoot=function(e,t){if(!sc(e))throw Error(A(299));var n=!1,r="",i=xh;return t!=null&&(t.unstable_strictMode===!0&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onRecoverableError!==void 0&&(i=t.onRecoverableError)),t=ic(e,1,!1,null,null,n,!1,r,i),e[en]=t.current,Ii(e.nodeType===8?e.parentNode:e),new oc(t)};ct.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var t=e._reactInternals;if(t===void 0)throw typeof e.render=="function"?Error(A(188)):(e=Object.keys(e).join(","),Error(A(268,e)));return e=qf(t),e=e===null?null:e.stateNode,e};ct.flushSync=function(e){return Gn(e)};ct.hydrate=function(e,t,n){if(!Vo(t))throw Error(A(200));return Uo(null,e,t,!0,n)};ct.hydrateRoot=function(e,t,n){if(!sc(e))throw Error(A(405));var r=n!=null&&n.hydratedSources||null,i=!1,a="",o=xh;if(n!=null&&(n.unstable_strictMode===!0&&(i=!0),n.identifierPrefix!==void 0&&(a=n.identifierPrefix),n.onRecoverableError!==void 0&&(o=n.onRecoverableError)),t=yh(t,null,e,1,n??null,i,!1,a,o),e[en]=t.current,Ii(e),r)for(e=0;e<r.length;e++)n=r[e],i=n._getVersion,i=i(n._source),t.mutableSourceEagerHydrationData==null?t.mutableSourceEagerHydrationData=[n,i]:t.mutableSourceEagerHydrationData.push(n,i);return new Bo(t)};ct.render=function(e,t,n){if(!Vo(t))throw Error(A(200));return Uo(null,e,t,!1,n)};ct.unmountComponentAtNode=function(e){if(!Vo(e))throw Error(A(40));return e._reactRootContainer?(Gn(function(){Uo(null,null,e,!1,function(){e._reactRootContainer=null,e[en]=null})}),!0):!1};ct.unstable_batchedUpdates=ec;ct.unstable_renderSubtreeIntoContainer=function(e,t,n,r){if(!Vo(n))throw Error(A(200));if(e==null||e._reactInternals===void 0)throw Error(A(38));return Uo(e,t,n,!1,r)};ct.version="18.3.1-next-f1338f8080-20240426";function _h(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(_h)}catch(e){console.error(e)}}_h(),_f.exports=ct;var Vx=_f.exports,qd=Vx;Js.createRoot=qd.createRoot,Js.hydrateRoot=qd.hydrateRoot;/**
 * @remix-run/router v1.23.2
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Ui(){return Ui=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Ui.apply(this,arguments)}var mn;(function(e){e.Pop="POP",e.Push="PUSH",e.Replace="REPLACE"})(mn||(mn={}));const Wd="popstate";function Ux(e){e===void 0&&(e={});function t(r,i){let{pathname:a,search:o,hash:s}=r.location;return $l("",{pathname:a,search:o,hash:s},i.state&&i.state.usr||null,i.state&&i.state.key||"default")}function n(r,i){return typeof i=="string"?i:wo(i)}return $x(t,n,null,e)}function be(e,t){if(e===!1||e===null||typeof e>"u")throw new Error(t)}function lc(e,t){if(!e){typeof console<"u"&&console.warn(t);try{throw new Error(t)}catch{}}}function Hx(){return Math.random().toString(36).substr(2,8)}function Bd(e,t){return{usr:e.state,key:e.key,idx:t}}function $l(e,t,n,r){return n===void 0&&(n=null),Ui({pathname:typeof e=="string"?e:e.pathname,search:"",hash:""},typeof t=="string"?Ur(t):t,{state:n,key:t&&t.key||r||Hx()})}function wo(e){let{pathname:t="/",search:n="",hash:r=""}=e;return n&&n!=="?"&&(t+=n.charAt(0)==="?"?n:"?"+n),r&&r!=="#"&&(t+=r.charAt(0)==="#"?r:"#"+r),t}function Ur(e){let t={};if(e){let n=e.indexOf("#");n>=0&&(t.hash=e.substr(n),e=e.substr(0,n));let r=e.indexOf("?");r>=0&&(t.search=e.substr(r),e=e.substr(0,r)),e&&(t.pathname=e)}return t}function $x(e,t,n,r){r===void 0&&(r={});let{window:i=document.defaultView,v5Compat:a=!1}=r,o=i.history,s=mn.Pop,l=null,u=d();u==null&&(u=0,o.replaceState(Ui({},o.state,{idx:u}),""));function d(){return(o.state||{idx:null}).idx}function c(){s=mn.Pop;let k=d(),h=k==null?null:k-u;u=k,l&&l({action:s,location:_.location,delta:h})}function f(k,h){s=mn.Push;let g=$l(_.location,k,h);u=d()+1;let x=Bd(g,u),C=_.createHref(g);try{o.pushState(x,"",C)}catch(z){if(z instanceof DOMException&&z.name==="DataCloneError")throw z;i.location.assign(C)}a&&l&&l({action:s,location:_.location,delta:1})}function m(k,h){s=mn.Replace;let g=$l(_.location,k,h);u=d();let x=Bd(g,u),C=_.createHref(g);o.replaceState(x,"",C),a&&l&&l({action:s,location:_.location,delta:0})}function y(k){let h=i.location.origin!=="null"?i.location.origin:i.location.href,g=typeof k=="string"?k:wo(k);return g=g.replace(/ $/,"%20"),be(h,"No window.location.(origin|href) available to create URL for href: "+g),new URL(g,h)}let _={get action(){return s},get location(){return e(i,o)},listen(k){if(l)throw new Error("A history only accepts one active listener");return i.addEventListener(Wd,c),l=k,()=>{i.removeEventListener(Wd,c),l=null}},createHref(k){return t(i,k)},createURL:y,encodeLocation(k){let h=y(k);return{pathname:h.pathname,search:h.search,hash:h.hash}},push:f,replace:m,go(k){return o.go(k)}};return _}var Vd;(function(e){e.data="data",e.deferred="deferred",e.redirect="redirect",e.error="error"})(Vd||(Vd={}));function Qx(e,t,n){return n===void 0&&(n="/"),Kx(e,t,n)}function Kx(e,t,n,r){let i=typeof t=="string"?Ur(t):t,a=Fr(i.pathname||"/",n);if(a==null)return null;let o=vh(e);Yx(o);let s=null;for(let l=0;s==null&&l<o.length;++l){let u=o1(a);s=i1(o[l],u)}return s}function vh(e,t,n,r){t===void 0&&(t=[]),n===void 0&&(n=[]),r===void 0&&(r="");let i=(a,o,s)=>{let l={relativePath:s===void 0?a.path||"":s,caseSensitive:a.caseSensitive===!0,childrenIndex:o,route:a};l.relativePath.startsWith("/")&&(be(l.relativePath.startsWith(r),'Absolute route path "'+l.relativePath+'" nested under path '+('"'+r+'" is not valid. An absolute child route path ')+"must start with the combined path of all its parent routes."),l.relativePath=l.relativePath.slice(r.length));let u=kn([r,l.relativePath]),d=n.concat(l);a.children&&a.children.length>0&&(be(a.index!==!0,"Index routes must not have child routes. Please remove "+('all child routes from route path "'+u+'".')),vh(a.children,t,d,u)),!(a.path==null&&!a.index)&&t.push({path:u,score:n1(u,a.index),routesMeta:d})};return e.forEach((a,o)=>{var s;if(a.path===""||!((s=a.path)!=null&&s.includes("?")))i(a,o);else for(let l of bh(a.path))i(a,o,l)}),t}function bh(e){let t=e.split("/");if(t.length===0)return[];let[n,...r]=t,i=n.endsWith("?"),a=n.replace(/\?$/,"");if(r.length===0)return i?[a,""]:[a];let o=bh(r.join("/")),s=[];return s.push(...o.map(l=>l===""?a:[a,l].join("/"))),i&&s.push(...o),s.map(l=>e.startsWith("/")&&l===""?"/":l)}function Yx(e){e.sort((t,n)=>t.score!==n.score?n.score-t.score:r1(t.routesMeta.map(r=>r.childrenIndex),n.routesMeta.map(r=>r.childrenIndex)))}const Xx=/^:[\w-]+$/,Gx=3,Zx=2,Jx=1,e1=10,t1=-2,Ud=e=>e==="*";function n1(e,t){let n=e.split("/"),r=n.length;return n.some(Ud)&&(r+=t1),t&&(r+=Zx),n.filter(i=>!Ud(i)).reduce((i,a)=>i+(Xx.test(a)?Gx:a===""?Jx:e1),r)}function r1(e,t){return e.length===t.length&&e.slice(0,-1).every((r,i)=>r===t[i])?e[e.length-1]-t[t.length-1]:0}function i1(e,t,n){let{routesMeta:r}=e,i={},a="/",o=[];for(let s=0;s<r.length;++s){let l=r[s],u=s===r.length-1,d=a==="/"?t:t.slice(a.length)||"/",c=Ql({path:l.relativePath,caseSensitive:l.caseSensitive,end:u},d),f=l.route;if(!c)return null;Object.assign(i,c.params),o.push({params:i,pathname:kn([a,c.pathname]),pathnameBase:d1(kn([a,c.pathnameBase])),route:f}),c.pathnameBase!=="/"&&(a=kn([a,c.pathnameBase]))}return o}function Ql(e,t){typeof e=="string"&&(e={path:e,caseSensitive:!1,end:!0});let[n,r]=a1(e.path,e.caseSensitive,e.end),i=t.match(n);if(!i)return null;let a=i[0],o=a.replace(/(.)\/+$/,"$1"),s=i.slice(1);return{params:r.reduce((u,d,c)=>{let{paramName:f,isOptional:m}=d;if(f==="*"){let _=s[c]||"";o=a.slice(0,a.length-_.length).replace(/(.)\/+$/,"$1")}const y=s[c];return m&&!y?u[f]=void 0:u[f]=(y||"").replace(/%2F/g,"/"),u},{}),pathname:a,pathnameBase:o,pattern:e}}function a1(e,t,n){t===void 0&&(t=!1),n===void 0&&(n=!0),lc(e==="*"||!e.endsWith("*")||e.endsWith("/*"),'Route path "'+e+'" will be treated as if it were '+('"'+e.replace(/\*$/,"/*")+'" because the `*` character must ')+"always follow a `/` in the pattern. To get rid of this warning, "+('please change the route path to "'+e.replace(/\*$/,"/*")+'".'));let r=[],i="^"+e.replace(/\/*\*?$/,"").replace(/^\/*/,"/").replace(/[\\.*+^${}|()[\]]/g,"\\$&").replace(/\/:([\w-]+)(\?)?/g,(o,s,l)=>(r.push({paramName:s,isOptional:l!=null}),l?"/?([^\\/]+)?":"/([^\\/]+)"));return e.endsWith("*")?(r.push({paramName:"*"}),i+=e==="*"||e==="/*"?"(.*)$":"(?:\\/(.+)|\\/*)$"):n?i+="\\/*$":e!==""&&e!=="/"&&(i+="(?:(?=\\/|$))"),[new RegExp(i,t?void 0:"i"),r]}function o1(e){try{return e.split("/").map(t=>decodeURIComponent(t).replace(/\//g,"%2F")).join("/")}catch(t){return lc(!1,'The URL path "'+e+'" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent '+("encoding ("+t+").")),e}}function Fr(e,t){if(t==="/")return e;if(!e.toLowerCase().startsWith(t.toLowerCase()))return null;let n=t.endsWith("/")?t.length-1:t.length,r=e.charAt(n);return r&&r!=="/"?null:e.slice(n)||"/"}const s1=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,l1=e=>s1.test(e);function u1(e,t){t===void 0&&(t="/");let{pathname:n,search:r="",hash:i=""}=typeof e=="string"?Ur(e):e,a;if(n)if(l1(n))a=n;else{if(n.includes("//")){let o=n;n=n.replace(/\/\/+/g,"/"),lc(!1,"Pathnames cannot have embedded double slashes - normalizing "+(o+" -> "+n))}n.startsWith("/")?a=Hd(n.substring(1),"/"):a=Hd(n,t)}else a=t;return{pathname:a,search:p1(r),hash:f1(i)}}function Hd(e,t){let n=t.replace(/\/+$/,"").split("/");return e.split("/").forEach(i=>{i===".."?n.length>1&&n.pop():i!=="."&&n.push(i)}),n.length>1?n.join("/"):"/"}function zs(e,t,n,r){return"Cannot include a '"+e+"' character in a manually specified "+("`to."+t+"` field ["+JSON.stringify(r)+"].  Please separate it out to the ")+("`to."+n+"` field. Alternatively you may provide the full path as ")+'a string in <Link to="..."> and the router will parse it for you.'}function c1(e){return e.filter((t,n)=>n===0||t.route.path&&t.route.path.length>0)}function wh(e,t){let n=c1(e);return t?n.map((r,i)=>i===n.length-1?r.pathname:r.pathnameBase):n.map(r=>r.pathnameBase)}function kh(e,t,n,r){r===void 0&&(r=!1);let i;typeof e=="string"?i=Ur(e):(i=Ui({},e),be(!i.pathname||!i.pathname.includes("?"),zs("?","pathname","search",i)),be(!i.pathname||!i.pathname.includes("#"),zs("#","pathname","hash",i)),be(!i.search||!i.search.includes("#"),zs("#","search","hash",i)));let a=e===""||i.pathname==="",o=a?"/":i.pathname,s;if(o==null)s=n;else{let c=t.length-1;if(!r&&o.startsWith("..")){let f=o.split("/");for(;f[0]==="..";)f.shift(),c-=1;i.pathname=f.join("/")}s=c>=0?t[c]:"/"}let l=u1(i,s),u=o&&o!=="/"&&o.endsWith("/"),d=(a||o===".")&&n.endsWith("/");return!l.pathname.endsWith("/")&&(u||d)&&(l.pathname+="/"),l}const kn=e=>e.join("/").replace(/\/\/+/g,"/"),d1=e=>e.replace(/\/+$/,"").replace(/^\/*/,"/"),p1=e=>!e||e==="?"?"":e.startsWith("?")?e:"?"+e,f1=e=>!e||e==="#"?"":e.startsWith("#")?e:"#"+e;function m1(e){return e!=null&&typeof e.status=="number"&&typeof e.statusText=="string"&&typeof e.internal=="boolean"&&"data"in e}const Sh=["post","put","patch","delete"];new Set(Sh);const h1=["get",...Sh];new Set(h1);/**
 * React Router v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Hi(){return Hi=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Hi.apply(this,arguments)}const Ho=E.createContext(null),Ch=E.createContext(null),Nn=E.createContext(null),$o=E.createContext(null),jn=E.createContext({outlet:null,matches:[],isDataRoute:!1}),Eh=E.createContext(null);function g1(e,t){let{relative:n}=t===void 0?{}:t;ta()||be(!1);let{basename:r,navigator:i}=E.useContext(Nn),{hash:a,pathname:o,search:s}=Qo(e,{relative:n}),l=o;return r!=="/"&&(l=o==="/"?r:kn([r,o])),i.createHref({pathname:l,search:s,hash:a})}function ta(){return E.useContext($o)!=null}function na(){return ta()||be(!1),E.useContext($o).location}function zh(e){E.useContext(Nn).static||E.useLayoutEffect(e)}function y1(){let{isDataRoute:e}=E.useContext(jn);return e?N1():x1()}function x1(){ta()||be(!1);let e=E.useContext(Ho),{basename:t,future:n,navigator:r}=E.useContext(Nn),{matches:i}=E.useContext(jn),{pathname:a}=na(),o=JSON.stringify(wh(i,n.v7_relativeSplatPath)),s=E.useRef(!1);return zh(()=>{s.current=!0}),E.useCallback(function(u,d){if(d===void 0&&(d={}),!s.current)return;if(typeof u=="number"){r.go(u);return}let c=kh(u,JSON.parse(o),a,d.relative==="path");e==null&&t!=="/"&&(c.pathname=c.pathname==="/"?t:kn([t,c.pathname])),(d.replace?r.replace:r.push)(c,d.state,d)},[t,r,o,a,e])}function Th(){let{matches:e}=E.useContext(jn),t=e[e.length-1];return t?t.params:{}}function Qo(e,t){let{relative:n}=t===void 0?{}:t,{future:r}=E.useContext(Nn),{matches:i}=E.useContext(jn),{pathname:a}=na(),o=JSON.stringify(wh(i,r.v7_relativeSplatPath));return E.useMemo(()=>kh(e,JSON.parse(o),a,n==="path"),[e,o,a,n])}function _1(e,t){return v1(e,t)}function v1(e,t,n,r){ta()||be(!1);let{navigator:i}=E.useContext(Nn),{matches:a}=E.useContext(jn),o=a[a.length-1],s=o?o.params:{};o&&o.pathname;let l=o?o.pathnameBase:"/";o&&o.route;let u=na(),d;if(t){var c;let k=typeof t=="string"?Ur(t):t;l==="/"||(c=k.pathname)!=null&&c.startsWith(l)||be(!1),d=k}else d=u;let f=d.pathname||"/",m=f;if(l!=="/"){let k=l.replace(/^\//,"").split("/");m="/"+f.replace(/^\//,"").split("/").slice(k.length).join("/")}let y=Qx(e,{pathname:m}),_=C1(y&&y.map(k=>Object.assign({},k,{params:Object.assign({},s,k.params),pathname:kn([l,i.encodeLocation?i.encodeLocation(k.pathname).pathname:k.pathname]),pathnameBase:k.pathnameBase==="/"?l:kn([l,i.encodeLocation?i.encodeLocation(k.pathnameBase).pathname:k.pathnameBase])})),a,n,r);return t&&_?E.createElement($o.Provider,{value:{location:Hi({pathname:"/",search:"",hash:"",state:null,key:"default"},d),navigationType:mn.Pop}},_):_}function b1(){let e=P1(),t=m1(e)?e.status+" "+e.statusText:e instanceof Error?e.message:JSON.stringify(e),n=e instanceof Error?e.stack:null,i={padding:"0.5rem",backgroundColor:"rgba(200,200,200, 0.5)"};return E.createElement(E.Fragment,null,E.createElement("h2",null,"Unexpected Application Error!"),E.createElement("h3",{style:{fontStyle:"italic"}},t),n?E.createElement("pre",{style:i},n):null,null)}const w1=E.createElement(b1,null);class k1 extends E.Component{constructor(t){super(t),this.state={location:t.location,revalidation:t.revalidation,error:t.error}}static getDerivedStateFromError(t){return{error:t}}static getDerivedStateFromProps(t,n){return n.location!==t.location||n.revalidation!=="idle"&&t.revalidation==="idle"?{error:t.error,location:t.location,revalidation:t.revalidation}:{error:t.error!==void 0?t.error:n.error,location:n.location,revalidation:t.revalidation||n.revalidation}}componentDidCatch(t,n){console.error("React Router caught the following error during render",t,n)}render(){return this.state.error!==void 0?E.createElement(jn.Provider,{value:this.props.routeContext},E.createElement(Eh.Provider,{value:this.state.error,children:this.props.component})):this.props.children}}function S1(e){let{routeContext:t,match:n,children:r}=e,i=E.useContext(Ho);return i&&i.static&&i.staticContext&&(n.route.errorElement||n.route.ErrorBoundary)&&(i.staticContext._deepestRenderedBoundaryId=n.route.id),E.createElement(jn.Provider,{value:t},r)}function C1(e,t,n,r){var i;if(t===void 0&&(t=[]),n===void 0&&(n=null),r===void 0&&(r=null),e==null){var a;if(!n)return null;if(n.errors)e=n.matches;else if((a=r)!=null&&a.v7_partialHydration&&t.length===0&&!n.initialized&&n.matches.length>0)e=n.matches;else return null}let o=e,s=(i=n)==null?void 0:i.errors;if(s!=null){let d=o.findIndex(c=>c.route.id&&(s==null?void 0:s[c.route.id])!==void 0);d>=0||be(!1),o=o.slice(0,Math.min(o.length,d+1))}let l=!1,u=-1;if(n&&r&&r.v7_partialHydration)for(let d=0;d<o.length;d++){let c=o[d];if((c.route.HydrateFallback||c.route.hydrateFallbackElement)&&(u=d),c.route.id){let{loaderData:f,errors:m}=n,y=c.route.loader&&f[c.route.id]===void 0&&(!m||m[c.route.id]===void 0);if(c.route.lazy||y){l=!0,u>=0?o=o.slice(0,u+1):o=[o[0]];break}}}return o.reduceRight((d,c,f)=>{let m,y=!1,_=null,k=null;n&&(m=s&&c.route.id?s[c.route.id]:void 0,_=c.route.errorElement||w1,l&&(u<0&&f===0?(j1("route-fallback"),y=!0,k=null):u===f&&(y=!0,k=c.route.hydrateFallbackElement||null)));let h=t.concat(o.slice(0,f+1)),g=()=>{let x;return m?x=_:y?x=k:c.route.Component?x=E.createElement(c.route.Component,null):c.route.element?x=c.route.element:x=d,E.createElement(S1,{match:c,routeContext:{outlet:d,matches:h,isDataRoute:n!=null},children:x})};return n&&(c.route.ErrorBoundary||c.route.errorElement||f===0)?E.createElement(k1,{location:n.location,revalidation:n.revalidation,component:_,error:m,children:g(),routeContext:{outlet:null,matches:h,isDataRoute:!0}}):g()},null)}var Ph=function(e){return e.UseBlocker="useBlocker",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e}(Ph||{}),Nh=function(e){return e.UseBlocker="useBlocker",e.UseLoaderData="useLoaderData",e.UseActionData="useActionData",e.UseRouteError="useRouteError",e.UseNavigation="useNavigation",e.UseRouteLoaderData="useRouteLoaderData",e.UseMatches="useMatches",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e.UseRouteId="useRouteId",e}(Nh||{});function E1(e){let t=E.useContext(Ho);return t||be(!1),t}function z1(e){let t=E.useContext(Ch);return t||be(!1),t}function T1(e){let t=E.useContext(jn);return t||be(!1),t}function jh(e){let t=T1(),n=t.matches[t.matches.length-1];return n.route.id||be(!1),n.route.id}function P1(){var e;let t=E.useContext(Eh),n=z1(),r=jh();return t!==void 0?t:(e=n.errors)==null?void 0:e[r]}function N1(){let{router:e}=E1(Ph.UseNavigateStable),t=jh(Nh.UseNavigateStable),n=E.useRef(!1);return zh(()=>{n.current=!0}),E.useCallback(function(i,a){a===void 0&&(a={}),n.current&&(typeof i=="number"?e.navigate(i):e.navigate(i,Hi({fromRouteId:t},a)))},[e,t])}const $d={};function j1(e,t,n){$d[e]||($d[e]=!0)}function L1(e,t){e==null||e.v7_startTransition,e==null||e.v7_relativeSplatPath}function Mn(e){be(!1)}function A1(e){let{basename:t="/",children:n=null,location:r,navigationType:i=mn.Pop,navigator:a,static:o=!1,future:s}=e;ta()&&be(!1);let l=t.replace(/^\/*/,"/"),u=E.useMemo(()=>({basename:l,navigator:a,static:o,future:Hi({v7_relativeSplatPath:!1},s)}),[l,s,a,o]);typeof r=="string"&&(r=Ur(r));let{pathname:d="/",search:c="",hash:f="",state:m=null,key:y="default"}=r,_=E.useMemo(()=>{let k=Fr(d,l);return k==null?null:{location:{pathname:k,search:c,hash:f,state:m,key:y},navigationType:i}},[l,d,c,f,m,y,i]);return _==null?null:E.createElement(Nn.Provider,{value:u},E.createElement($o.Provider,{children:n,value:_}))}function R1(e){let{children:t,location:n}=e;return _1(Kl(t),n)}new Promise(()=>{});function Kl(e,t){t===void 0&&(t=[]);let n=[];return E.Children.forEach(e,(r,i)=>{if(!E.isValidElement(r))return;let a=[...t,i];if(r.type===E.Fragment){n.push.apply(n,Kl(r.props.children,a));return}r.type!==Mn&&be(!1),!r.props.index||!r.props.children||be(!1);let o={id:r.props.id||a.join("-"),caseSensitive:r.props.caseSensitive,element:r.props.element,Component:r.props.Component,index:r.props.index,path:r.props.path,loader:r.props.loader,action:r.props.action,errorElement:r.props.errorElement,ErrorBoundary:r.props.ErrorBoundary,hasErrorBoundary:r.props.ErrorBoundary!=null||r.props.errorElement!=null,shouldRevalidate:r.props.shouldRevalidate,handle:r.props.handle,lazy:r.props.lazy};r.props.children&&(o.children=Kl(r.props.children,a)),n.push(o)}),n}/**
 * React Router DOM v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function ko(){return ko=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},ko.apply(this,arguments)}function Lh(e,t){if(e==null)return{};var n={},r=Object.keys(e),i,a;for(a=0;a<r.length;a++)i=r[a],!(t.indexOf(i)>=0)&&(n[i]=e[i]);return n}function I1(e){return!!(e.metaKey||e.altKey||e.ctrlKey||e.shiftKey)}function O1(e,t){return e.button===0&&(!t||t==="_self")&&!I1(e)}const D1=["onClick","relative","reloadDocument","replace","state","target","to","preventScrollReset","viewTransition"],F1=["aria-current","caseSensitive","className","end","style","to","viewTransition","children"],M1="6";try{window.__reactRouterVersion=M1}catch{}const q1=E.createContext({isTransitioning:!1}),W1="startTransition",Qd=Ly[W1];function B1(e){let{basename:t,children:n,future:r,window:i}=e,a=E.useRef();a.current==null&&(a.current=Ux({window:i,v5Compat:!0}));let o=a.current,[s,l]=E.useState({action:o.action,location:o.location}),{v7_startTransition:u}=r||{},d=E.useCallback(c=>{u&&Qd?Qd(()=>l(c)):l(c)},[l,u]);return E.useLayoutEffect(()=>o.listen(d),[o,d]),E.useEffect(()=>L1(r),[r]),E.createElement(A1,{basename:t,children:n,location:s.location,navigationType:s.action,navigator:o,future:r})}const V1=typeof window<"u"&&typeof window.document<"u"&&typeof window.document.createElement<"u",U1=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,ot=E.forwardRef(function(t,n){let{onClick:r,relative:i,reloadDocument:a,replace:o,state:s,target:l,to:u,preventScrollReset:d,viewTransition:c}=t,f=Lh(t,D1),{basename:m}=E.useContext(Nn),y,_=!1;if(typeof u=="string"&&U1.test(u)&&(y=u,V1))try{let x=new URL(window.location.href),C=u.startsWith("//")?new URL(x.protocol+u):new URL(u),z=Fr(C.pathname,m);C.origin===x.origin&&z!=null?u=z+C.search+C.hash:_=!0}catch{}let k=g1(u,{relative:i}),h=Q1(u,{replace:o,state:s,target:l,preventScrollReset:d,relative:i,viewTransition:c});function g(x){r&&r(x),x.defaultPrevented||h(x)}return E.createElement("a",ko({},f,{href:y||k,onClick:_||a?r:g,ref:n,target:l}))}),H1=E.forwardRef(function(t,n){let{"aria-current":r="page",caseSensitive:i=!1,className:a="",end:o=!1,style:s,to:l,viewTransition:u,children:d}=t,c=Lh(t,F1),f=Qo(l,{relative:c.relative}),m=na(),y=E.useContext(Ch),{navigator:_,basename:k}=E.useContext(Nn),h=y!=null&&K1(f)&&u===!0,g=_.encodeLocation?_.encodeLocation(f).pathname:f.pathname,x=m.pathname,C=y&&y.navigation&&y.navigation.location?y.navigation.location.pathname:null;i||(x=x.toLowerCase(),C=C?C.toLowerCase():null,g=g.toLowerCase()),C&&k&&(C=Fr(C,k)||C);const z=g!=="/"&&g.endsWith("/")?g.length-1:g.length;let S=x===g||!o&&x.startsWith(g)&&x.charAt(z)==="/",T=C!=null&&(C===g||!o&&C.startsWith(g)&&C.charAt(g.length)==="/"),N={isActive:S,isPending:T,isTransitioning:h},M=S?r:void 0,w;typeof a=="function"?w=a(N):w=[a,S?"active":null,T?"pending":null,h?"transitioning":null].filter(Boolean).join(" ");let I=typeof s=="function"?s(N):s;return E.createElement(ot,ko({},c,{"aria-current":M,className:w,ref:n,style:I,to:l,viewTransition:u}),typeof d=="function"?d(N):d)});var Yl;(function(e){e.UseScrollRestoration="useScrollRestoration",e.UseSubmit="useSubmit",e.UseSubmitFetcher="useSubmitFetcher",e.UseFetcher="useFetcher",e.useViewTransitionState="useViewTransitionState"})(Yl||(Yl={}));var Kd;(function(e){e.UseFetcher="useFetcher",e.UseFetchers="useFetchers",e.UseScrollRestoration="useScrollRestoration"})(Kd||(Kd={}));function $1(e){let t=E.useContext(Ho);return t||be(!1),t}function Q1(e,t){let{target:n,replace:r,state:i,preventScrollReset:a,relative:o,viewTransition:s}=t===void 0?{}:t,l=y1(),u=na(),d=Qo(e,{relative:o});return E.useCallback(c=>{if(O1(c,n)){c.preventDefault();let f=r!==void 0?r:wo(u)===wo(d);l(e,{replace:f,state:i,preventScrollReset:a,relative:o,viewTransition:s})}},[u,l,d,r,i,n,e,a,o,s])}function K1(e,t){t===void 0&&(t={});let n=E.useContext(q1);n==null&&be(!1);let{basename:r}=$1(Yl.useViewTransitionState),i=Qo(e,{relative:t.relative});if(!n.isTransitioning)return!1;let a=Fr(n.currentLocation.pathname,r)||n.currentLocation.pathname,o=Fr(n.nextLocation.pathname,r)||n.nextLocation.pathname;return Ql(i.pathname,o)!=null||Ql(i.pathname,a)!=null}var Y1=typeof Element<"u",X1=typeof Map=="function",G1=typeof Set=="function",Z1=typeof ArrayBuffer=="function"&&!!ArrayBuffer.isView;function $a(e,t){if(e===t)return!0;if(e&&t&&typeof e=="object"&&typeof t=="object"){if(e.constructor!==t.constructor)return!1;var n,r,i;if(Array.isArray(e)){if(n=e.length,n!=t.length)return!1;for(r=n;r--!==0;)if(!$a(e[r],t[r]))return!1;return!0}var a;if(X1&&e instanceof Map&&t instanceof Map){if(e.size!==t.size)return!1;for(a=e.entries();!(r=a.next()).done;)if(!t.has(r.value[0]))return!1;for(a=e.entries();!(r=a.next()).done;)if(!$a(r.value[1],t.get(r.value[0])))return!1;return!0}if(G1&&e instanceof Set&&t instanceof Set){if(e.size!==t.size)return!1;for(a=e.entries();!(r=a.next()).done;)if(!t.has(r.value[0]))return!1;return!0}if(Z1&&ArrayBuffer.isView(e)&&ArrayBuffer.isView(t)){if(n=e.length,n!=t.length)return!1;for(r=n;r--!==0;)if(e[r]!==t[r])return!1;return!0}if(e.constructor===RegExp)return e.source===t.source&&e.flags===t.flags;if(e.valueOf!==Object.prototype.valueOf&&typeof e.valueOf=="function"&&typeof t.valueOf=="function")return e.valueOf()===t.valueOf();if(e.toString!==Object.prototype.toString&&typeof e.toString=="function"&&typeof t.toString=="function")return e.toString()===t.toString();if(i=Object.keys(e),n=i.length,n!==Object.keys(t).length)return!1;for(r=n;r--!==0;)if(!Object.prototype.hasOwnProperty.call(t,i[r]))return!1;if(Y1&&e instanceof Element)return!1;for(r=n;r--!==0;)if(!((i[r]==="_owner"||i[r]==="__v"||i[r]==="__o")&&e.$$typeof)&&!$a(e[i[r]],t[i[r]]))return!1;return!0}return e!==e&&t!==t}var J1=function(t,n){try{return $a(t,n)}catch(r){if((r.message||"").match(/stack|recursion/i))return console.warn("react-fast-compare cannot handle circular refs"),!1;throw r}};const e_=Jn(J1);var t_=function(e,t,n,r,i,a,o,s){if(!e){var l;if(t===void 0)l=new Error("Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.");else{var u=[n,r,i,a,o,s],d=0;l=new Error(t.replace(/%s/g,function(){return u[d++]})),l.name="Invariant Violation"}throw l.framesToPop=1,l}},n_=t_;const Yd=Jn(n_);var r_=function(t,n,r,i){var a=r?r.call(i,t,n):void 0;if(a!==void 0)return!!a;if(t===n)return!0;if(typeof t!="object"||!t||typeof n!="object"||!n)return!1;var o=Object.keys(t),s=Object.keys(n);if(o.length!==s.length)return!1;for(var l=Object.prototype.hasOwnProperty.bind(n),u=0;u<o.length;u++){var d=o[u];if(!l(d))return!1;var c=t[d],f=n[d];if(a=r?r.call(i,c,f,d):void 0,a===!1||a===void 0&&c!==f)return!1}return!0};const i_=Jn(r_);var Ah=(e=>(e.BASE="base",e.BODY="body",e.HEAD="head",e.HTML="html",e.LINK="link",e.META="meta",e.NOSCRIPT="noscript",e.SCRIPT="script",e.STYLE="style",e.TITLE="title",e.FRAGMENT="Symbol(react.fragment)",e))(Ah||{}),Ts={link:{rel:["amphtml","canonical","alternate"]},script:{type:["application/ld+json"]},meta:{charset:"",name:["generator","robots","description"],property:["og:type","og:title","og:url","og:image","og:image:alt","og:description","twitter:url","twitter:title","twitter:description","twitter:image","twitter:image:alt","twitter:card","twitter:site"]}},Xd=Object.values(Ah),uc={accesskey:"accessKey",charset:"charSet",class:"className",contenteditable:"contentEditable",contextmenu:"contextMenu","http-equiv":"httpEquiv",itemprop:"itemProp",tabindex:"tabIndex"},a_=Object.entries(uc).reduce((e,[t,n])=>(e[n]=t,e),{}),jt="data-rh",zr={DEFAULT_TITLE:"defaultTitle",DEFER:"defer",ENCODE_SPECIAL_CHARACTERS:"encodeSpecialCharacters",ON_CHANGE_CLIENT_STATE:"onChangeClientState",TITLE_TEMPLATE:"titleTemplate",PRIORITIZE_SEO_TAGS:"prioritizeSeoTags"},Tr=(e,t)=>{for(let n=e.length-1;n>=0;n-=1){const r=e[n];if(Object.prototype.hasOwnProperty.call(r,t))return r[t]}return null},o_=e=>{let t=Tr(e,"title");const n=Tr(e,zr.TITLE_TEMPLATE);if(Array.isArray(t)&&(t=t.join("")),n&&t)return n.replace(/%s/g,()=>t);const r=Tr(e,zr.DEFAULT_TITLE);return t||r||void 0},s_=e=>Tr(e,zr.ON_CHANGE_CLIENT_STATE)||(()=>{}),Ps=(e,t)=>t.filter(n=>typeof n[e]<"u").map(n=>n[e]).reduce((n,r)=>({...n,...r}),{}),l_=(e,t)=>t.filter(n=>typeof n.base<"u").map(n=>n.base).reverse().reduce((n,r)=>{if(!n.length){const i=Object.keys(r);for(let a=0;a<i.length;a+=1){const s=i[a].toLowerCase();if(e.indexOf(s)!==-1&&r[s])return n.concat(r)}}return n},[]),u_=e=>console&&typeof console.warn=="function"&&console.warn(e),ii=(e,t,n)=>{const r={};return n.filter(i=>Array.isArray(i[e])?!0:(typeof i[e]<"u"&&u_(`Helmet: ${e} should be of type "Array". Instead found type "${typeof i[e]}"`),!1)).map(i=>i[e]).reverse().reduce((i,a)=>{const o={};a.filter(l=>{let u;const d=Object.keys(l);for(let f=0;f<d.length;f+=1){const m=d[f],y=m.toLowerCase();t.indexOf(y)!==-1&&!(u==="rel"&&l[u].toLowerCase()==="canonical")&&!(y==="rel"&&l[y].toLowerCase()==="stylesheet")&&(u=y),t.indexOf(m)!==-1&&(m==="innerHTML"||m==="cssText"||m==="itemprop")&&(u=m)}if(!u||!l[u])return!1;const c=l[u].toLowerCase();return r[u]||(r[u]={}),o[u]||(o[u]={}),r[u][c]?!1:(o[u][c]=!0,!0)}).reverse().forEach(l=>i.push(l));const s=Object.keys(o);for(let l=0;l<s.length;l+=1){const u=s[l],d={...r[u],...o[u]};r[u]=d}return i},[]).reverse()},c_=(e,t)=>{if(Array.isArray(e)&&e.length){for(let n=0;n<e.length;n+=1)if(e[n][t])return!0}return!1},d_=e=>({baseTag:l_(["href"],e),bodyAttributes:Ps("bodyAttributes",e),defer:Tr(e,zr.DEFER),encode:Tr(e,zr.ENCODE_SPECIAL_CHARACTERS),htmlAttributes:Ps("htmlAttributes",e),linkTags:ii("link",["rel","href"],e),metaTags:ii("meta",["name","charset","http-equiv","property","itemprop"],e),noscriptTags:ii("noscript",["innerHTML"],e),onChangeClientState:s_(e),scriptTags:ii("script",["src","innerHTML"],e),styleTags:ii("style",["cssText"],e),title:o_(e),titleAttributes:Ps("titleAttributes",e),prioritizeSeoTags:c_(e,zr.PRIORITIZE_SEO_TAGS)}),Rh=e=>Array.isArray(e)?e.join(""):e,p_=(e,t)=>{const n=Object.keys(e);for(let r=0;r<n.length;r+=1)if(t[n[r]]&&t[n[r]].includes(e[n[r]]))return!0;return!1},Ns=(e,t)=>Array.isArray(e)?e.reduce((n,r)=>(p_(r,t)?n.priority.push(r):n.default.push(r),n),{priority:[],default:[]}):{default:e,priority:[]},Gd=(e,t)=>({...e,[t]:void 0}),f_=["noscript","script","style"],Xl=(e,t=!0)=>t===!1?String(e):String(e).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#x27;"),Ih=e=>Object.keys(e).reduce((t,n)=>{const r=typeof e[n]<"u"?`${n}="${e[n]}"`:`${n}`;return t?`${t} ${r}`:r},""),m_=(e,t,n,r)=>{const i=Ih(n),a=Rh(t);return i?`<${e} ${jt}="true" ${i}>${Xl(a,r)}</${e}>`:`<${e} ${jt}="true">${Xl(a,r)}</${e}>`},h_=(e,t,n=!0)=>t.reduce((r,i)=>{const a=i,o=Object.keys(a).filter(u=>!(u==="innerHTML"||u==="cssText")).reduce((u,d)=>{const c=typeof a[d]>"u"?d:`${d}="${Xl(a[d],n)}"`;return u?`${u} ${c}`:c},""),s=a.innerHTML||a.cssText||"",l=f_.indexOf(e)===-1;return`${r}<${e} ${jt}="true" ${o}${l?"/>":`>${s}</${e}>`}`},""),Oh=(e,t={})=>Object.keys(e).reduce((n,r)=>{const i=uc[r];return n[i||r]=e[r],n},t),g_=(e,t,n)=>{const r={key:t,[jt]:!0},i=Oh(n,r);return[Ne.createElement("title",i,t)]},Qa=(e,t)=>t.map((n,r)=>{const i={key:r,[jt]:!0};return Object.keys(n).forEach(a=>{const s=uc[a]||a;if(s==="innerHTML"||s==="cssText"){const l=n.innerHTML||n.cssText;i.dangerouslySetInnerHTML={__html:l}}else i[s]=n[a]}),Ne.createElement(e,i)}),mt=(e,t,n=!0)=>{switch(e){case"title":return{toComponent:()=>g_(e,t.title,t.titleAttributes),toString:()=>m_(e,t.title,t.titleAttributes,n)};case"bodyAttributes":case"htmlAttributes":return{toComponent:()=>Oh(t),toString:()=>Ih(t)};default:return{toComponent:()=>Qa(e,t),toString:()=>h_(e,t,n)}}},y_=({metaTags:e,linkTags:t,scriptTags:n,encode:r})=>{const i=Ns(e,Ts.meta),a=Ns(t,Ts.link),o=Ns(n,Ts.script);return{priorityMethods:{toComponent:()=>[...Qa("meta",i.priority),...Qa("link",a.priority),...Qa("script",o.priority)],toString:()=>`${mt("meta",i.priority,r)} ${mt("link",a.priority,r)} ${mt("script",o.priority,r)}`},metaTags:i.default,linkTags:a.default,scriptTags:o.default}},x_=e=>{const{baseTag:t,bodyAttributes:n,encode:r=!0,htmlAttributes:i,noscriptTags:a,styleTags:o,title:s="",titleAttributes:l,prioritizeSeoTags:u}=e;let{linkTags:d,metaTags:c,scriptTags:f}=e,m={toComponent:()=>{},toString:()=>""};return u&&({priorityMethods:m,linkTags:d,metaTags:c,scriptTags:f}=y_(e)),{priority:m,base:mt("base",t,r),bodyAttributes:mt("bodyAttributes",n,r),htmlAttributes:mt("htmlAttributes",i,r),link:mt("link",d,r),meta:mt("meta",c,r),noscript:mt("noscript",a,r),script:mt("script",f,r),style:mt("style",o,r),title:mt("title",{title:s,titleAttributes:l},r)}},Gl=x_,Ca=[],Dh=!!(typeof window<"u"&&window.document&&window.document.createElement),Zl=class{constructor(e,t){$t(this,"instances",[]);$t(this,"canUseDOM",Dh);$t(this,"context");$t(this,"value",{setHelmet:e=>{this.context.helmet=e},helmetInstances:{get:()=>this.canUseDOM?Ca:this.instances,add:e=>{(this.canUseDOM?Ca:this.instances).push(e)},remove:e=>{const t=(this.canUseDOM?Ca:this.instances).indexOf(e);(this.canUseDOM?Ca:this.instances).splice(t,1)}}});this.context=e,this.canUseDOM=t||!1,t||(e.helmet=Gl({baseTag:[],bodyAttributes:{},htmlAttributes:{},linkTags:[],metaTags:[],noscriptTags:[],scriptTags:[],styleTags:[],title:"",titleAttributes:{}}))}},__={},Fh=Ne.createContext(__),Un,Mh=(Un=class extends E.Component{constructor(n){super(n);$t(this,"helmetData");this.helmetData=new Zl(this.props.context||{},Un.canUseDOM)}render(){return Ne.createElement(Fh.Provider,{value:this.helmetData.value},this.props.children)}},$t(Un,"canUseDOM",Dh),Un),ar=(e,t)=>{const n=document.head||document.querySelector("head"),r=n.querySelectorAll(`${e}[${jt}]`),i=[].slice.call(r),a=[];let o;return t&&t.length&&t.forEach(s=>{const l=document.createElement(e);for(const u in s)if(Object.prototype.hasOwnProperty.call(s,u))if(u==="innerHTML")l.innerHTML=s.innerHTML;else if(u==="cssText")l.styleSheet?l.styleSheet.cssText=s.cssText:l.appendChild(document.createTextNode(s.cssText));else{const d=u,c=typeof s[d]>"u"?"":s[d];l.setAttribute(u,c)}l.setAttribute(jt,"true"),i.some((u,d)=>(o=d,l.isEqualNode(u)))?i.splice(o,1):a.push(l)}),i.forEach(s=>{var l;return(l=s.parentNode)==null?void 0:l.removeChild(s)}),a.forEach(s=>n.appendChild(s)),{oldTags:i,newTags:a}},Jl=(e,t)=>{const n=document.getElementsByTagName(e)[0];if(!n)return;const r=n.getAttribute(jt),i=r?r.split(","):[],a=[...i],o=Object.keys(t);for(const s of o){const l=t[s]||"";n.getAttribute(s)!==l&&n.setAttribute(s,l),i.indexOf(s)===-1&&i.push(s);const u=a.indexOf(s);u!==-1&&a.splice(u,1)}for(let s=a.length-1;s>=0;s-=1)n.removeAttribute(a[s]);i.length===a.length?n.removeAttribute(jt):n.getAttribute(jt)!==o.join(",")&&n.setAttribute(jt,o.join(","))},v_=(e,t)=>{typeof e<"u"&&document.title!==e&&(document.title=Rh(e)),Jl("title",t)},Zd=(e,t)=>{const{baseTag:n,bodyAttributes:r,htmlAttributes:i,linkTags:a,metaTags:o,noscriptTags:s,onChangeClientState:l,scriptTags:u,styleTags:d,title:c,titleAttributes:f}=e;Jl("body",r),Jl("html",i),v_(c,f);const m={baseTag:ar("base",n),linkTags:ar("link",a),metaTags:ar("meta",o),noscriptTags:ar("noscript",s),scriptTags:ar("script",u),styleTags:ar("style",d)},y={},_={};Object.keys(m).forEach(k=>{const{newTags:h,oldTags:g}=m[k];h.length&&(y[k]=h),g.length&&(_[k]=m[k].oldTags)}),t&&t(),l(e,y,_)},ai=null,b_=e=>{ai&&cancelAnimationFrame(ai),e.defer?ai=requestAnimationFrame(()=>{Zd(e,()=>{ai=null})}):(Zd(e),ai=null)},w_=b_,Jd=class extends E.Component{constructor(){super(...arguments);$t(this,"rendered",!1)}shouldComponentUpdate(t){return!i_(t,this.props)}componentDidUpdate(){this.emitChange()}componentWillUnmount(){const{helmetInstances:t}=this.props.context;t.remove(this),this.emitChange()}emitChange(){const{helmetInstances:t,setHelmet:n}=this.props.context;let r=null;const i=d_(t.get().map(a=>{const o={...a.props};return delete o.context,o}));Mh.canUseDOM?w_(i):Gl&&(r=Gl(i)),n(r)}init(){if(this.rendered)return;this.rendered=!0;const{helmetInstances:t}=this.props.context;t.add(this),this.emitChange()}render(){return this.init(),null}},Zs,k_=(Zs=class extends E.Component{shouldComponentUpdate(e){return!e_(Gd(this.props,"helmetData"),Gd(e,"helmetData"))}mapNestedChildrenToProps(e,t){if(!t)return null;switch(e.type){case"script":case"noscript":return{innerHTML:t};case"style":return{cssText:t};default:throw new Error(`<${e.type} /> elements are self-closing and can not contain children. Refer to our API for more information.`)}}flattenArrayTypeChildren(e,t,n,r){return{...t,[e.type]:[...t[e.type]||[],{...n,...this.mapNestedChildrenToProps(e,r)}]}}mapObjectTypeChildren(e,t,n,r){switch(e.type){case"title":return{...t,[e.type]:r,titleAttributes:{...n}};case"body":return{...t,bodyAttributes:{...n}};case"html":return{...t,htmlAttributes:{...n}};default:return{...t,[e.type]:{...n}}}}mapArrayTypeChildrenToProps(e,t){let n={...t};return Object.keys(e).forEach(r=>{n={...n,[r]:e[r]}}),n}warnOnInvalidChildren(e,t){return Yd(Xd.some(n=>e.type===n),typeof e.type=="function"?"You may be attempting to nest <Helmet> components within each other, which is not allowed. Refer to our API for more information.":`Only elements types ${Xd.join(", ")} are allowed. Helmet does not support rendering <${e.type}> elements. Refer to our API for more information.`),Yd(!t||typeof t=="string"||Array.isArray(t)&&!t.some(n=>typeof n!="string"),`Helmet expects a string as a child of <${e.type}>. Did you forget to wrap your children in braces? ( <${e.type}>{\`\`}</${e.type}> ) Refer to our API for more information.`),!0}mapChildrenToProps(e,t){let n={};return Ne.Children.forEach(e,r=>{if(!r||!r.props)return;const{children:i,...a}=r.props,o=Object.keys(a).reduce((l,u)=>(l[a_[u]||u]=a[u],l),{});let{type:s}=r;switch(typeof s=="symbol"?s=s.toString():this.warnOnInvalidChildren(r,i),s){case"Symbol(react.fragment)":t=this.mapChildrenToProps(i,t);break;case"link":case"meta":case"noscript":case"script":case"style":n=this.flattenArrayTypeChildren(r,n,o,i);break;default:t=this.mapObjectTypeChildren(r,t,o,i);break}}),this.mapArrayTypeChildrenToProps(n,t)}render(){const{children:e,...t}=this.props;let n={...t},{helmetData:r}=t;if(e&&(n=this.mapChildrenToProps(e,n)),r&&!(r instanceof Zl)){const i=r;r=new Zl(i.context,!0),delete n.helmetData}return r?Ne.createElement(Jd,{...n,context:r.value}):Ne.createElement(Fh.Consumer,null,i=>Ne.createElement(Jd,{...n,context:i}))}},$t(Zs,"defaultProps",{defer:!0,encodeSpecialCharacters:!0,prioritizeSeoTags:!1}),Zs);const js="ml-interview-progress",qh=E.createContext(null);function S_({children:e}){const[t,n]=E.useState(()=>{const u=localStorage.getItem(js);return u?JSON.parse(u):{}});E.useEffect(()=>{localStorage.setItem(js,JSON.stringify(t))},[t]);const r=E.useCallback((u,d,c)=>{n(f=>{var m;return{...f,[u]:{...f[u],[d]:{...(m=f[u])==null?void 0:m[d],status:c,lastAttempt:new Date().toISOString()}}}})},[]),i=E.useCallback((u,d,c)=>{n(f=>{var m,y,_;return{...f,[u]:{...f[u],[d]:{...(m=f[u])==null?void 0:m[d],status:((_=(y=f[u])==null?void 0:y[d])==null?void 0:_.status)||"in_progress",code:c,lastAttempt:new Date().toISOString()}}}})},[]),a=E.useCallback((u,d)=>{var f;const c=(f=t[u])==null?void 0:f[d];return{status:(c==null?void 0:c.status)||"not_started",code:c==null?void 0:c.code}},[t]),o=E.useCallback((u,d)=>{const c=t[u];if(!c||d===0)return 0;const f=Object.values(c).filter(m=>m.status==="completed").length;return Math.round(f/d*100)},[t]),s=E.useCallback(u=>{const d=u.reduce((f,m)=>f+m.problemCount,0);if(d===0)return 0;let c=0;for(const f of u){const m=t[f.id];m&&(c+=Object.values(m).filter(y=>y.status==="completed").length)}return Math.round(c/d*100)},[t]),l=E.useCallback(()=>{n({}),localStorage.removeItem(js)},[]);return p.jsx(qh.Provider,{value:{progress:t,updateProblemStatus:r,saveProblemCode:i,getProblemProgress:a,getSectionProgress:o,getOverallProgress:s,resetProgress:l},children:e})}function ra(){const e=E.useContext(qh);if(!e)throw new Error("useProgress must be used within a ProgressProvider");return e}const Mr=[{id:"numpy-fundamentals",title:"NumPy Fundamentals",description:"Master NumPy array operations essential for ML development.",icon:"🔢",introduction:"\n# NumPy Fundamentals\n\nNumPy is the backbone of scientific computing in Python. Every ML library builds on it.\n\n## Key Concepts\n\n### Array Creation\n- `np.array()`: Create from lists\n- `np.zeros()`, `np.ones()`: Initialize with values\n- `np.arange()`, `np.linspace()`: Numeric ranges\n- `np.eye()`: Identity matrix\n\n### Indexing & Slicing\n- Basic: `arr[0]`, `arr[1:5]`\n- Advanced: `arr[[0, 2, 4]]`, `arr[arr > 0]`\n- Multi-dimensional: `arr[1, :]`, `arr[:, 2:4]`\n\n### Broadcasting\nArrays of different shapes can be combined:\n- (3, 4) + (4,) → broadcasts to (3, 4)\n- (3, 1) * (1, 4) → broadcasts to (3, 4)\n\n### Aggregations\n- `sum()`, `mean()`, `std()` - with axis parameter\n- `argmax()`, `argmin()` - find indices\n\n### Shape Manipulation\n- `reshape()`: Change dimensions\n- `transpose()`, `.T`: Swap axes\n- `flatten()`, `ravel()`: To 1D\n\nMaster these operations to write efficient ML code!\n    ",problems:["numpy-array-creation","numpy-indexing","numpy-broadcasting","numpy-aggregations","numpy-reshape-transpose"]},{id:"python-basics",title:"Python Basics for ML",description:"Master NumPy arrays and essential Python operations for machine learning.",icon:"🐍",introduction:"\n# Python Basics for Machine Learning\n\nNumPy is the foundation of nearly all machine learning in Python. Before diving into complex algorithms, you need to master array operations.\n\n## Key Concepts\n\n### NumPy Arrays\n- **Creation**: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`\n- **Shape manipulation**: `reshape()`, `flatten()`, `transpose()`\n- **Indexing**: Slicing, boolean indexing, fancy indexing\n\n### Essential Operations\n- **Element-wise**: Addition, multiplication, mathematical functions\n- **Aggregations**: `sum()`, `mean()`, `std()`, `min()`, `max()`\n- **Broadcasting**: Automatic expansion of arrays for operations\n\n### Why It Matters\nEvery ML algorithm processes data as arrays. Understanding these operations lets you:\n- Prepare data efficiently\n- Debug model issues\n- Optimize performance\n\nLet's practice with hands-on problems!\n    ",problems:["numpy-array-sum","numpy-matrix-multiply","numpy-broadcast-add"]},{id:"einsum",title:"Einstein Summation (Einsum)",description:"Master einsum for elegant tensor operations.",icon:"∑",introduction:`
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
    `,problems:["kl-divergence","vae-reparameterization","vae-loss","vqvae-quantization","diffusion-noise-schedule","diffusion-forward"]},{id:"reinforcement-learning",title:"Reinforcement Learning",description:"Master RL fundamentals from Q-learning to policy gradients.",icon:"🎮",introduction:`
# Reinforcement Learning

Reinforcement learning (RL) trains agents to make decisions by learning from rewards and punishments through trial and error.

## Key Concepts

### The RL Framework
- **Agent**: The learner/decision-maker
- **Environment**: What the agent interacts with
- **State (s)**: Current situation
- **Action (a)**: What the agent can do
- **Reward (r)**: Feedback signal
- **Policy (π)**: Strategy for choosing actions

### Value Functions
- **State Value V(s)**: Expected return starting from state s
- **Action Value Q(s,a)**: Expected return taking action a in state s

### Bellman Equations
\`\`\`python
V(s) = E[r + γ * V(s')]
Q(s,a) = E[r + γ * max_a' Q(s',a')]
\`\`\`

### Key Algorithms

| Algorithm | Type | Policy |
|-----------|------|--------|
| Q-Learning | Value-based | Off-policy |
| SARSA | Value-based | On-policy |
| REINFORCE | Policy-based | On-policy |
| Actor-Critic | Hybrid | On-policy |
| PPO | Policy-based | On-policy |

### Exploration vs Exploitation
- **Exploration**: Try new actions to discover better strategies
- **Exploitation**: Use known good actions to maximize reward
- **ε-greedy**: Simple balance (random with probability ε)

### RL for LLMs (RLHF)
- **Reward Modeling**: Learn preferences from human feedback
- **PPO for LLMs**: Fine-tune with KL constraints
- **DPO**: Direct preference optimization without reward model

### Interview Essentials
- Implement Q-learning and SARSA from scratch
- Explain on-policy vs off-policy
- Understand policy gradients and REINFORCE
- Know GAE and PPO for modern deep RL
- **RLHF**: Reward modeling, KL penalties, DPO

Master these concepts for ML interviews at top companies!
    `,problems:["rl-discounted-return","rl-epsilon-greedy","rl-bellman-value","rl-q-learning-update","rl-sarsa-update","rl-td0-prediction","rl-value-iteration","rl-n-step-return","rl-policy-gradient","rl-advantage-estimation","rl-ppo-clip","rl-reward-modeling","rl-dpo-loss","rl-kl-penalty","rl-ppo-llm-objective"]},{id:"e2e-implementations",title:"End-to-End Implementations",description:"Build complete ML models from scratch.",icon:"🏗️",introduction:`
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
    `,problems:["e2e-mlp","e2e-transformer","e2e-vae","e2e-vqvae","e2e-diffusion","e2e-cnn"]}];function C_({isOpen:e,onClose:t}){const[n,r]=E.useState(""),[i,a]=E.useState(""),[o,s]=E.useState(!1),[l,u]=E.useState(!1),[d,c]=E.useState("");if(!e)return null;const f=async y=>{y.preventDefault(),s(!0),c("");try{(await fetch("https://formspree.io/f/mpqqwpav",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:n,feedback:i,_subject:`ML Coding Lab Feedback from ${n}`})})).ok?(u(!0),r(""),a("")):c("Failed to send feedback. Please try again.")}catch{c("Failed to send feedback. Please try again.")}finally{s(!1)}},m=()=>{u(!1),c(""),t()};return p.jsxs("div",{className:"fixed inset-0 z-50 flex items-center justify-center",children:[p.jsx("div",{className:"absolute inset-0 bg-black bg-opacity-50",onClick:m}),p.jsxs("div",{className:"relative bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-6",children:[p.jsx("button",{onClick:m,className:"absolute top-4 right-4 text-gray-400 hover:text-gray-600",children:p.jsx("svg",{className:"w-5 h-5",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M6 18L18 6M6 6l12 12"})})}),p.jsx("h2",{className:"text-xl font-semibold text-gray-900 mb-4",children:"Send Feedback"}),l?p.jsxs("div",{className:"text-center py-8",children:[p.jsx("div",{className:"w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4",children:p.jsx("svg",{className:"w-6 h-6 text-green-600",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M5 13l4 4L19 7"})})}),p.jsx("p",{className:"text-gray-700 mb-4",children:"Thank you for your feedback!"}),p.jsx("button",{onClick:m,className:"px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors",children:"Close"})]}):p.jsxs("form",{onSubmit:f,className:"space-y-4",children:[p.jsxs("div",{children:[p.jsx("label",{htmlFor:"name",className:"block text-sm font-medium text-gray-700 mb-1",children:"Name"}),p.jsx("input",{type:"text",id:"name",value:n,onChange:y=>r(y.target.value),required:!0,className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent",placeholder:"Your name"})]}),p.jsxs("div",{children:[p.jsx("label",{htmlFor:"feedback",className:"block text-sm font-medium text-gray-700 mb-1",children:"Feedback"}),p.jsx("textarea",{id:"feedback",value:i,onChange:y=>a(y.target.value),required:!0,rows:4,className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none",placeholder:"Share your thoughts, suggestions, or report issues..."})]}),d&&p.jsx("p",{className:"text-red-600 text-sm",children:d}),p.jsxs("div",{className:"flex gap-3 justify-end",children:[p.jsx("button",{type:"button",onClick:m,className:"px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors",children:"Cancel"}),p.jsx("button",{type:"submit",disabled:o,className:"px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:o?"Sending...":"Send Feedback"})]})]})]})]})}function E_(){const{getOverallProgress:e}=ra(),[t,n]=E.useState(!1),r=Mr.map(a=>({id:a.id,problemCount:a.problems.length})),i=e(r);return p.jsxs(p.Fragment,{children:[p.jsx("header",{className:"bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm",children:p.jsx("div",{className:"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",children:p.jsxs("div",{className:"flex items-center justify-between h-16",children:[p.jsxs(ot,{to:"/",className:"flex items-center gap-3",children:[p.jsx("img",{src:"/ml-coding-lab/vite.svg",alt:"ML Coding Lab",className:"w-8 h-8"}),p.jsx("span",{className:"text-xl font-semibold text-gray-900",children:"ML Coding Lab"})]}),p.jsxs("div",{className:"flex items-center gap-6",children:[p.jsxs(ot,{to:"/scratchpad",className:"px-3 py-1.5 bg-gray-100 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-200 transition-colors flex items-center gap-2",children:[p.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"})}),"Scratchpad"]}),p.jsxs("a",{href:"/ml-coding-lab/ml-cheatsheet.html",target:"_blank",rel:"noopener noreferrer",className:"px-3 py-1.5 bg-amber-100 text-amber-700 text-sm font-medium rounded-md hover:bg-amber-200 transition-colors flex items-center gap-2",children:[p.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"})}),"Cheat Sheet"]}),p.jsxs("button",{onClick:()=>n(!0),className:"px-3 py-1.5 text-gray-500 text-sm font-medium hover:text-gray-700 transition-colors flex items-center gap-1",children:[p.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"})}),"Feedback"]}),p.jsxs("div",{className:"flex items-center gap-3",children:[p.jsx("span",{className:"text-gray-500 text-sm",children:"Overall Progress"}),p.jsx("div",{className:"w-32 h-2 bg-gray-200 rounded-full overflow-hidden",children:p.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${i}%`}})}),p.jsxs("span",{className:"text-primary-600 text-sm font-medium",children:[i,"%"]})]})]})]})})}),p.jsx(C_,{isOpen:t,onClose:()=>n(!1)})]})}function z_(){const{getSectionProgress:e}=ra();return p.jsx("aside",{className:"w-64 bg-white border-r border-gray-200 min-h-[calc(100vh-4rem)] p-4",children:p.jsxs("nav",{className:"space-y-2",children:[p.jsx("h3",{className:"text-gray-500 text-xs font-semibold uppercase tracking-wider mb-4 px-3",children:"Learning Sections"}),Mr.map((t,n)=>{const r=e(t.id,t.problems.length);return p.jsxs(H1,{to:`/section/${t.id}`,className:({isActive:i})=>`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${i?"bg-primary-50 text-primary-700":"text-gray-700 hover:bg-gray-100 hover:text-gray-900"}`,children:[p.jsx("span",{className:"w-6 h-6 flex items-center justify-center text-lg",children:t.icon}),p.jsxs("div",{className:"flex-1 min-w-0",children:[p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsxs("span",{className:"text-gray-400 text-xs",children:[n+1,"."]}),p.jsx("span",{className:"text-sm font-medium truncate",children:t.title})]}),p.jsxs("div",{className:"flex items-center gap-2 mt-1",children:[p.jsx("div",{className:"flex-1 h-1 bg-gray-200 rounded-full overflow-hidden",children:p.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${r}%`}})}),p.jsxs("span",{className:"text-xs text-gray-400",children:[r,"%"]})]})]})]},t.id)})]})})}function T_(){const e=new Date().getFullYear();return p.jsx("footer",{className:"bg-white border-t border-gray-200 mt-auto",children:p.jsxs("div",{className:"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6",children:[p.jsxs("div",{className:"flex flex-col md:flex-row justify-between items-center gap-4",children:[p.jsx("div",{className:"flex items-center gap-2 text-gray-600 text-sm",children:p.jsxs("span",{children:["© ",e," Siddharth Choudhary. All rights reserved."]})}),p.jsxs("div",{className:"flex items-center gap-6 text-sm",children:[p.jsx(ot,{to:"/terms",className:"text-gray-500 hover:text-gray-700 transition-colors",children:"Terms of Service"}),p.jsx(ot,{to:"/privacy",className:"text-gray-500 hover:text-gray-700 transition-colors",children:"Privacy Policy"}),p.jsxs("a",{href:"https://github.com/itzsid/ml-coding-lab",target:"_blank",rel:"noopener noreferrer",className:"text-gray-500 hover:text-gray-700 transition-colors flex items-center gap-1",children:[p.jsx("svg",{className:"w-4 h-4",fill:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"})}),"GitHub"]})]})]}),p.jsx("div",{className:"mt-4 pt-4 border-t border-gray-100 text-center text-xs text-gray-400",children:p.jsx("p",{children:"The code for this project is licensed under the MIT License. Educational content is copyrighted and may not be reproduced without permission."})})]})})}function P_({children:e}){return p.jsxs("div",{className:"min-h-screen bg-gray-50 flex flex-col",children:[p.jsx(E_,{}),p.jsxs("div",{className:"flex flex-1",children:[p.jsx(z_,{}),p.jsx("main",{className:"flex-1 p-6",children:e})]}),p.jsx(T_,{})]})}const ep="https://itzsid.github.io/ml-coding-lab",N_="ML Coding Lab - Learn Machine Learning by Building",j_="Practice hands-on machine learning coding problems. Implement NumPy, neural networks, CNNs, transformers, and more. Run Python code directly in your browser with instant feedback.",L_="machine learning, ML, deep learning, neural networks, Python, NumPy, transformers, CNN, PyTorch, coding practice, interview prep, AI";function Hr({title:e,description:t=j_,keywords:n=L_,canonical:r,type:i="website"}){const a=e?`${e} | ML Coding Lab`:N_,o=r?`${ep}${r}`:ep;return p.jsxs(k_,{children:[p.jsx("title",{children:a}),p.jsx("meta",{name:"title",content:a}),p.jsx("meta",{name:"description",content:t}),p.jsx("meta",{name:"keywords",content:n}),p.jsx("link",{rel:"canonical",href:o}),p.jsx("meta",{property:"og:type",content:i}),p.jsx("meta",{property:"og:url",content:o}),p.jsx("meta",{property:"og:title",content:a}),p.jsx("meta",{property:"og:description",content:t}),p.jsx("meta",{property:"twitter:url",content:o}),p.jsx("meta",{property:"twitter:title",content:a}),p.jsx("meta",{property:"twitter:description",content:t})]})}function A_(){const{getSectionProgress:e,getOverallProgress:t}=ra(),n=Mr.map(i=>({id:i.id,problemCount:i.problems.length})),r=t(n);return p.jsxs("div",{className:"max-w-5xl mx-auto",children:[p.jsx(Hr,{canonical:"/"}),p.jsxs("div",{className:"text-center mb-12",children:[p.jsx("h1",{className:"text-4xl font-bold text-gray-900 mb-4",children:"Learn ML By Building"}),p.jsx("p",{className:"text-lg text-gray-600 max-w-2xl mx-auto mb-8",children:"Practice hands-on machine learning coding problems. Run Python code directly in your browser with NumPy and scikit-learn support."}),p.jsxs("div",{className:"inline-flex items-center gap-4 bg-white rounded-lg px-6 py-3 shadow-sm border border-gray-200",children:[p.jsx("span",{className:"text-gray-600",children:"Overall Progress"}),p.jsx("div",{className:"w-48 h-2 bg-gray-200 rounded-full overflow-hidden",children:p.jsx("div",{className:"h-full bg-primary-500 transition-all duration-500",style:{width:`${r}%`}})}),p.jsxs("span",{className:"text-primary-600 font-medium",children:[r,"%"]})]})]}),p.jsx("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:Mr.map((i,a)=>{const o=e(i.id,i.problems.length);return p.jsx(ot,{to:`/section/${i.id}`,className:"group bg-white rounded-xl p-6 border border-gray-200 hover:border-primary-400 hover:shadow-md transition-all duration-200",children:p.jsxs("div",{className:"flex items-start gap-4",children:[p.jsx("div",{className:"w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center text-2xl group-hover:bg-primary-50 transition-colors",children:i.icon}),p.jsxs("div",{className:"flex-1",children:[p.jsxs("div",{className:"flex items-center gap-2 mb-1",children:[p.jsxs("span",{className:"text-gray-400 text-sm",children:[a+1,"."]}),p.jsx("h3",{className:"text-lg font-semibold text-gray-900 group-hover:text-primary-600 transition-colors",children:i.title})]}),p.jsx("p",{className:"text-gray-500 text-sm mb-4",children:i.description}),p.jsxs("div",{className:"flex items-center justify-between",children:[p.jsxs("span",{className:"text-gray-400 text-sm",children:[i.problems.length," problems"]}),p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("div",{className:"w-24 h-1.5 bg-gray-200 rounded-full overflow-hidden",children:p.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${o}%`}})}),p.jsxs("span",{className:"text-xs text-gray-500",children:[o,"%"]})]})]})]})]})},i.id)})}),p.jsxs("div",{className:"mt-16 grid grid-cols-1 md:grid-cols-3 gap-6",children:[p.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[p.jsx("div",{className:"text-3xl mb-3",children:"🐍"}),p.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Python in Browser"}),p.jsx("p",{className:"text-gray-500 text-sm",children:"Run Python code with NumPy, pandas, and more - no setup required."})]}),p.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[p.jsx("div",{className:"text-3xl mb-3",children:"✅"}),p.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Instant Feedback"}),p.jsx("p",{className:"text-gray-500 text-sm",children:"Test your code against multiple test cases and see results immediately."})]}),p.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[p.jsx("div",{className:"text-3xl mb-3",children:"📊"}),p.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Track Progress"}),p.jsx("p",{className:"text-gray-500 text-sm",children:"Your progress is saved locally. Pick up where you left off anytime."})]})]})]})}function R_(e,t){const n={};return(e[e.length-1]===""?[...e,""]:e).join((n.padRight?" ":"")+","+(n.padLeft===!1?"":" ")).trim()}const I_=/^[$_\p{ID_Start}][$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,O_=/^[$_\p{ID_Start}][-$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,D_={};function tp(e,t){return(D_.jsx?O_:I_).test(e)}const F_=/[ \t\n\f\r]/g;function M_(e){return typeof e=="object"?e.type==="text"?np(e.value):!1:np(e)}function np(e){return e.replace(F_,"")===""}class ia{constructor(t,n,r){this.normal=n,this.property=t,r&&(this.space=r)}}ia.prototype.normal={};ia.prototype.property={};ia.prototype.space=void 0;function Wh(e,t){const n={},r={};for(const i of e)Object.assign(n,i.property),Object.assign(r,i.normal);return new ia(n,r,t)}function eu(e){return e.toLowerCase()}class et{constructor(t,n){this.attribute=n,this.property=t}}et.prototype.attribute="";et.prototype.booleanish=!1;et.prototype.boolean=!1;et.prototype.commaOrSpaceSeparated=!1;et.prototype.commaSeparated=!1;et.prototype.defined=!1;et.prototype.mustUseProperty=!1;et.prototype.number=!1;et.prototype.overloadedBoolean=!1;et.prototype.property="";et.prototype.spaceSeparated=!1;et.prototype.space=void 0;let q_=0;const G=nr(),Se=nr(),tu=nr(),R=nr(),de=nr(),Pr=nr(),nt=nr();function nr(){return 2**++q_}const nu=Object.freeze(Object.defineProperty({__proto__:null,boolean:G,booleanish:Se,commaOrSpaceSeparated:nt,commaSeparated:Pr,number:R,overloadedBoolean:tu,spaceSeparated:de},Symbol.toStringTag,{value:"Module"})),Ls=Object.keys(nu);class cc extends et{constructor(t,n,r,i){let a=-1;if(super(t,n),rp(this,"space",i),typeof r=="number")for(;++a<Ls.length;){const o=Ls[a];rp(this,Ls[a],(r&nu[o])===nu[o])}}}cc.prototype.defined=!0;function rp(e,t,n){n&&(e[t]=n)}function $r(e){const t={},n={};for(const[r,i]of Object.entries(e.properties)){const a=new cc(r,e.transform(e.attributes||{},r),i,e.space);e.mustUseProperty&&e.mustUseProperty.includes(r)&&(a.mustUseProperty=!0),t[r]=a,n[eu(r)]=r,n[eu(a.attribute)]=r}return new ia(t,n,e.space)}const Bh=$r({properties:{ariaActiveDescendant:null,ariaAtomic:Se,ariaAutoComplete:null,ariaBusy:Se,ariaChecked:Se,ariaColCount:R,ariaColIndex:R,ariaColSpan:R,ariaControls:de,ariaCurrent:null,ariaDescribedBy:de,ariaDetails:null,ariaDisabled:Se,ariaDropEffect:de,ariaErrorMessage:null,ariaExpanded:Se,ariaFlowTo:de,ariaGrabbed:Se,ariaHasPopup:null,ariaHidden:Se,ariaInvalid:null,ariaKeyShortcuts:null,ariaLabel:null,ariaLabelledBy:de,ariaLevel:R,ariaLive:null,ariaModal:Se,ariaMultiLine:Se,ariaMultiSelectable:Se,ariaOrientation:null,ariaOwns:de,ariaPlaceholder:null,ariaPosInSet:R,ariaPressed:Se,ariaReadOnly:Se,ariaRelevant:null,ariaRequired:Se,ariaRoleDescription:de,ariaRowCount:R,ariaRowIndex:R,ariaRowSpan:R,ariaSelected:Se,ariaSetSize:R,ariaSort:null,ariaValueMax:R,ariaValueMin:R,ariaValueNow:R,ariaValueText:null,role:null},transform(e,t){return t==="role"?t:"aria-"+t.slice(4).toLowerCase()}});function Vh(e,t){return t in e?e[t]:t}function Uh(e,t){return Vh(e,t.toLowerCase())}const W_=$r({attributes:{acceptcharset:"accept-charset",classname:"class",htmlfor:"for",httpequiv:"http-equiv"},mustUseProperty:["checked","multiple","muted","selected"],properties:{abbr:null,accept:Pr,acceptCharset:de,accessKey:de,action:null,allow:null,allowFullScreen:G,allowPaymentRequest:G,allowUserMedia:G,alt:null,as:null,async:G,autoCapitalize:null,autoComplete:de,autoFocus:G,autoPlay:G,blocking:de,capture:null,charSet:null,checked:G,cite:null,className:de,cols:R,colSpan:null,content:null,contentEditable:Se,controls:G,controlsList:de,coords:R|Pr,crossOrigin:null,data:null,dateTime:null,decoding:null,default:G,defer:G,dir:null,dirName:null,disabled:G,download:tu,draggable:Se,encType:null,enterKeyHint:null,fetchPriority:null,form:null,formAction:null,formEncType:null,formMethod:null,formNoValidate:G,formTarget:null,headers:de,height:R,hidden:tu,high:R,href:null,hrefLang:null,htmlFor:de,httpEquiv:de,id:null,imageSizes:null,imageSrcSet:null,inert:G,inputMode:null,integrity:null,is:null,isMap:G,itemId:null,itemProp:de,itemRef:de,itemScope:G,itemType:de,kind:null,label:null,lang:null,language:null,list:null,loading:null,loop:G,low:R,manifest:null,max:null,maxLength:R,media:null,method:null,min:null,minLength:R,multiple:G,muted:G,name:null,nonce:null,noModule:G,noValidate:G,onAbort:null,onAfterPrint:null,onAuxClick:null,onBeforeMatch:null,onBeforePrint:null,onBeforeToggle:null,onBeforeUnload:null,onBlur:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onContextLost:null,onContextMenu:null,onContextRestored:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnded:null,onError:null,onFocus:null,onFormData:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLanguageChange:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadEnd:null,onLoadStart:null,onMessage:null,onMessageError:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRejectionHandled:null,onReset:null,onResize:null,onScroll:null,onScrollEnd:null,onSecurityPolicyViolation:null,onSeeked:null,onSeeking:null,onSelect:null,onSlotChange:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnhandledRejection:null,onUnload:null,onVolumeChange:null,onWaiting:null,onWheel:null,open:G,optimum:R,pattern:null,ping:de,placeholder:null,playsInline:G,popover:null,popoverTarget:null,popoverTargetAction:null,poster:null,preload:null,readOnly:G,referrerPolicy:null,rel:de,required:G,reversed:G,rows:R,rowSpan:R,sandbox:de,scope:null,scoped:G,seamless:G,selected:G,shadowRootClonable:G,shadowRootDelegatesFocus:G,shadowRootMode:null,shape:null,size:R,sizes:null,slot:null,span:R,spellCheck:Se,src:null,srcDoc:null,srcLang:null,srcSet:null,start:R,step:null,style:null,tabIndex:R,target:null,title:null,translate:null,type:null,typeMustMatch:G,useMap:null,value:Se,width:R,wrap:null,writingSuggestions:null,align:null,aLink:null,archive:de,axis:null,background:null,bgColor:null,border:R,borderColor:null,bottomMargin:R,cellPadding:null,cellSpacing:null,char:null,charOff:null,classId:null,clear:null,code:null,codeBase:null,codeType:null,color:null,compact:G,declare:G,event:null,face:null,frame:null,frameBorder:null,hSpace:R,leftMargin:R,link:null,longDesc:null,lowSrc:null,marginHeight:R,marginWidth:R,noResize:G,noHref:G,noShade:G,noWrap:G,object:null,profile:null,prompt:null,rev:null,rightMargin:R,rules:null,scheme:null,scrolling:Se,standby:null,summary:null,text:null,topMargin:R,valueType:null,version:null,vAlign:null,vLink:null,vSpace:R,allowTransparency:null,autoCorrect:null,autoSave:null,disablePictureInPicture:G,disableRemotePlayback:G,prefix:null,property:null,results:R,security:null,unselectable:null},space:"html",transform:Uh}),B_=$r({attributes:{accentHeight:"accent-height",alignmentBaseline:"alignment-baseline",arabicForm:"arabic-form",baselineShift:"baseline-shift",capHeight:"cap-height",className:"class",clipPath:"clip-path",clipRule:"clip-rule",colorInterpolation:"color-interpolation",colorInterpolationFilters:"color-interpolation-filters",colorProfile:"color-profile",colorRendering:"color-rendering",crossOrigin:"crossorigin",dataType:"datatype",dominantBaseline:"dominant-baseline",enableBackground:"enable-background",fillOpacity:"fill-opacity",fillRule:"fill-rule",floodColor:"flood-color",floodOpacity:"flood-opacity",fontFamily:"font-family",fontSize:"font-size",fontSizeAdjust:"font-size-adjust",fontStretch:"font-stretch",fontStyle:"font-style",fontVariant:"font-variant",fontWeight:"font-weight",glyphName:"glyph-name",glyphOrientationHorizontal:"glyph-orientation-horizontal",glyphOrientationVertical:"glyph-orientation-vertical",hrefLang:"hreflang",horizAdvX:"horiz-adv-x",horizOriginX:"horiz-origin-x",horizOriginY:"horiz-origin-y",imageRendering:"image-rendering",letterSpacing:"letter-spacing",lightingColor:"lighting-color",markerEnd:"marker-end",markerMid:"marker-mid",markerStart:"marker-start",navDown:"nav-down",navDownLeft:"nav-down-left",navDownRight:"nav-down-right",navLeft:"nav-left",navNext:"nav-next",navPrev:"nav-prev",navRight:"nav-right",navUp:"nav-up",navUpLeft:"nav-up-left",navUpRight:"nav-up-right",onAbort:"onabort",onActivate:"onactivate",onAfterPrint:"onafterprint",onBeforePrint:"onbeforeprint",onBegin:"onbegin",onCancel:"oncancel",onCanPlay:"oncanplay",onCanPlayThrough:"oncanplaythrough",onChange:"onchange",onClick:"onclick",onClose:"onclose",onCopy:"oncopy",onCueChange:"oncuechange",onCut:"oncut",onDblClick:"ondblclick",onDrag:"ondrag",onDragEnd:"ondragend",onDragEnter:"ondragenter",onDragExit:"ondragexit",onDragLeave:"ondragleave",onDragOver:"ondragover",onDragStart:"ondragstart",onDrop:"ondrop",onDurationChange:"ondurationchange",onEmptied:"onemptied",onEnd:"onend",onEnded:"onended",onError:"onerror",onFocus:"onfocus",onFocusIn:"onfocusin",onFocusOut:"onfocusout",onHashChange:"onhashchange",onInput:"oninput",onInvalid:"oninvalid",onKeyDown:"onkeydown",onKeyPress:"onkeypress",onKeyUp:"onkeyup",onLoad:"onload",onLoadedData:"onloadeddata",onLoadedMetadata:"onloadedmetadata",onLoadStart:"onloadstart",onMessage:"onmessage",onMouseDown:"onmousedown",onMouseEnter:"onmouseenter",onMouseLeave:"onmouseleave",onMouseMove:"onmousemove",onMouseOut:"onmouseout",onMouseOver:"onmouseover",onMouseUp:"onmouseup",onMouseWheel:"onmousewheel",onOffline:"onoffline",onOnline:"ononline",onPageHide:"onpagehide",onPageShow:"onpageshow",onPaste:"onpaste",onPause:"onpause",onPlay:"onplay",onPlaying:"onplaying",onPopState:"onpopstate",onProgress:"onprogress",onRateChange:"onratechange",onRepeat:"onrepeat",onReset:"onreset",onResize:"onresize",onScroll:"onscroll",onSeeked:"onseeked",onSeeking:"onseeking",onSelect:"onselect",onShow:"onshow",onStalled:"onstalled",onStorage:"onstorage",onSubmit:"onsubmit",onSuspend:"onsuspend",onTimeUpdate:"ontimeupdate",onToggle:"ontoggle",onUnload:"onunload",onVolumeChange:"onvolumechange",onWaiting:"onwaiting",onZoom:"onzoom",overlinePosition:"overline-position",overlineThickness:"overline-thickness",paintOrder:"paint-order",panose1:"panose-1",pointerEvents:"pointer-events",referrerPolicy:"referrerpolicy",renderingIntent:"rendering-intent",shapeRendering:"shape-rendering",stopColor:"stop-color",stopOpacity:"stop-opacity",strikethroughPosition:"strikethrough-position",strikethroughThickness:"strikethrough-thickness",strokeDashArray:"stroke-dasharray",strokeDashOffset:"stroke-dashoffset",strokeLineCap:"stroke-linecap",strokeLineJoin:"stroke-linejoin",strokeMiterLimit:"stroke-miterlimit",strokeOpacity:"stroke-opacity",strokeWidth:"stroke-width",tabIndex:"tabindex",textAnchor:"text-anchor",textDecoration:"text-decoration",textRendering:"text-rendering",transformOrigin:"transform-origin",typeOf:"typeof",underlinePosition:"underline-position",underlineThickness:"underline-thickness",unicodeBidi:"unicode-bidi",unicodeRange:"unicode-range",unitsPerEm:"units-per-em",vAlphabetic:"v-alphabetic",vHanging:"v-hanging",vIdeographic:"v-ideographic",vMathematical:"v-mathematical",vectorEffect:"vector-effect",vertAdvY:"vert-adv-y",vertOriginX:"vert-origin-x",vertOriginY:"vert-origin-y",wordSpacing:"word-spacing",writingMode:"writing-mode",xHeight:"x-height",playbackOrder:"playbackorder",timelineBegin:"timelinebegin"},properties:{about:nt,accentHeight:R,accumulate:null,additive:null,alignmentBaseline:null,alphabetic:R,amplitude:R,arabicForm:null,ascent:R,attributeName:null,attributeType:null,azimuth:R,bandwidth:null,baselineShift:null,baseFrequency:null,baseProfile:null,bbox:null,begin:null,bias:R,by:null,calcMode:null,capHeight:R,className:de,clip:null,clipPath:null,clipPathUnits:null,clipRule:null,color:null,colorInterpolation:null,colorInterpolationFilters:null,colorProfile:null,colorRendering:null,content:null,contentScriptType:null,contentStyleType:null,crossOrigin:null,cursor:null,cx:null,cy:null,d:null,dataType:null,defaultAction:null,descent:R,diffuseConstant:R,direction:null,display:null,dur:null,divisor:R,dominantBaseline:null,download:G,dx:null,dy:null,edgeMode:null,editable:null,elevation:R,enableBackground:null,end:null,event:null,exponent:R,externalResourcesRequired:null,fill:null,fillOpacity:R,fillRule:null,filter:null,filterRes:null,filterUnits:null,floodColor:null,floodOpacity:null,focusable:null,focusHighlight:null,fontFamily:null,fontSize:null,fontSizeAdjust:null,fontStretch:null,fontStyle:null,fontVariant:null,fontWeight:null,format:null,fr:null,from:null,fx:null,fy:null,g1:Pr,g2:Pr,glyphName:Pr,glyphOrientationHorizontal:null,glyphOrientationVertical:null,glyphRef:null,gradientTransform:null,gradientUnits:null,handler:null,hanging:R,hatchContentUnits:null,hatchUnits:null,height:null,href:null,hrefLang:null,horizAdvX:R,horizOriginX:R,horizOriginY:R,id:null,ideographic:R,imageRendering:null,initialVisibility:null,in:null,in2:null,intercept:R,k:R,k1:R,k2:R,k3:R,k4:R,kernelMatrix:nt,kernelUnitLength:null,keyPoints:null,keySplines:null,keyTimes:null,kerning:null,lang:null,lengthAdjust:null,letterSpacing:null,lightingColor:null,limitingConeAngle:R,local:null,markerEnd:null,markerMid:null,markerStart:null,markerHeight:null,markerUnits:null,markerWidth:null,mask:null,maskContentUnits:null,maskUnits:null,mathematical:null,max:null,media:null,mediaCharacterEncoding:null,mediaContentEncodings:null,mediaSize:R,mediaTime:null,method:null,min:null,mode:null,name:null,navDown:null,navDownLeft:null,navDownRight:null,navLeft:null,navNext:null,navPrev:null,navRight:null,navUp:null,navUpLeft:null,navUpRight:null,numOctaves:null,observer:null,offset:null,onAbort:null,onActivate:null,onAfterPrint:null,onBeforePrint:null,onBegin:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnd:null,onEnded:null,onError:null,onFocus:null,onFocusIn:null,onFocusOut:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadStart:null,onMessage:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onMouseWheel:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRepeat:null,onReset:null,onResize:null,onScroll:null,onSeeked:null,onSeeking:null,onSelect:null,onShow:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnload:null,onVolumeChange:null,onWaiting:null,onZoom:null,opacity:null,operator:null,order:null,orient:null,orientation:null,origin:null,overflow:null,overlay:null,overlinePosition:R,overlineThickness:R,paintOrder:null,panose1:null,path:null,pathLength:R,patternContentUnits:null,patternTransform:null,patternUnits:null,phase:null,ping:de,pitch:null,playbackOrder:null,pointerEvents:null,points:null,pointsAtX:R,pointsAtY:R,pointsAtZ:R,preserveAlpha:null,preserveAspectRatio:null,primitiveUnits:null,propagate:null,property:nt,r:null,radius:null,referrerPolicy:null,refX:null,refY:null,rel:nt,rev:nt,renderingIntent:null,repeatCount:null,repeatDur:null,requiredExtensions:nt,requiredFeatures:nt,requiredFonts:nt,requiredFormats:nt,resource:null,restart:null,result:null,rotate:null,rx:null,ry:null,scale:null,seed:null,shapeRendering:null,side:null,slope:null,snapshotTime:null,specularConstant:R,specularExponent:R,spreadMethod:null,spacing:null,startOffset:null,stdDeviation:null,stemh:null,stemv:null,stitchTiles:null,stopColor:null,stopOpacity:null,strikethroughPosition:R,strikethroughThickness:R,string:null,stroke:null,strokeDashArray:nt,strokeDashOffset:null,strokeLineCap:null,strokeLineJoin:null,strokeMiterLimit:R,strokeOpacity:R,strokeWidth:null,style:null,surfaceScale:R,syncBehavior:null,syncBehaviorDefault:null,syncMaster:null,syncTolerance:null,syncToleranceDefault:null,systemLanguage:nt,tabIndex:R,tableValues:null,target:null,targetX:R,targetY:R,textAnchor:null,textDecoration:null,textRendering:null,textLength:null,timelineBegin:null,title:null,transformBehavior:null,type:null,typeOf:nt,to:null,transform:null,transformOrigin:null,u1:null,u2:null,underlinePosition:R,underlineThickness:R,unicode:null,unicodeBidi:null,unicodeRange:null,unitsPerEm:R,values:null,vAlphabetic:R,vMathematical:R,vectorEffect:null,vHanging:R,vIdeographic:R,version:null,vertAdvY:R,vertOriginX:R,vertOriginY:R,viewBox:null,viewTarget:null,visibility:null,width:null,widths:null,wordSpacing:null,writingMode:null,x:null,x1:null,x2:null,xChannelSelector:null,xHeight:R,y:null,y1:null,y2:null,yChannelSelector:null,z:null,zoomAndPan:null},space:"svg",transform:Vh}),Hh=$r({properties:{xLinkActuate:null,xLinkArcRole:null,xLinkHref:null,xLinkRole:null,xLinkShow:null,xLinkTitle:null,xLinkType:null},space:"xlink",transform(e,t){return"xlink:"+t.slice(5).toLowerCase()}}),$h=$r({attributes:{xmlnsxlink:"xmlns:xlink"},properties:{xmlnsXLink:null,xmlns:null},space:"xmlns",transform:Uh}),Qh=$r({properties:{xmlBase:null,xmlLang:null,xmlSpace:null},space:"xml",transform(e,t){return"xml:"+t.slice(3).toLowerCase()}}),V_={classId:"classID",dataType:"datatype",itemId:"itemID",strokeDashArray:"strokeDasharray",strokeDashOffset:"strokeDashoffset",strokeLineCap:"strokeLinecap",strokeLineJoin:"strokeLinejoin",strokeMiterLimit:"strokeMiterlimit",typeOf:"typeof",xLinkActuate:"xlinkActuate",xLinkArcRole:"xlinkArcrole",xLinkHref:"xlinkHref",xLinkRole:"xlinkRole",xLinkShow:"xlinkShow",xLinkTitle:"xlinkTitle",xLinkType:"xlinkType",xmlnsXLink:"xmlnsXlink"},U_=/[A-Z]/g,ip=/-[a-z]/g,H_=/^data[-\w.:]+$/i;function $_(e,t){const n=eu(t);let r=t,i=et;if(n in e.normal)return e.property[e.normal[n]];if(n.length>4&&n.slice(0,4)==="data"&&H_.test(t)){if(t.charAt(4)==="-"){const a=t.slice(5).replace(ip,K_);r="data"+a.charAt(0).toUpperCase()+a.slice(1)}else{const a=t.slice(4);if(!ip.test(a)){let o=a.replace(U_,Q_);o.charAt(0)!=="-"&&(o="-"+o),t="data"+o}}i=cc}return new i(r,t)}function Q_(e){return"-"+e.toLowerCase()}function K_(e){return e.charAt(1).toUpperCase()}const Y_=Wh([Bh,W_,Hh,$h,Qh],"html"),dc=Wh([Bh,B_,Hh,$h,Qh],"svg");function X_(e){return e.join(" ").trim()}var pc={},ap=/\/\*[^*]*\*+([^/*][^*]*\*+)*\//g,G_=/\n/g,Z_=/^\s*/,J_=/^(\*?[-#/*\\\w]+(\[[0-9a-z_-]+\])?)\s*/,ev=/^:\s*/,tv=/^((?:'(?:\\'|.)*?'|"(?:\\"|.)*?"|\([^)]*?\)|[^};])+)/,nv=/^[;\s]*/,rv=/^\s+|\s+$/g,iv=`
`,op="/",sp="*",qn="",av="comment",ov="declaration";function sv(e,t){if(typeof e!="string")throw new TypeError("First argument must be a string");if(!e)return[];t=t||{};var n=1,r=1;function i(y){var _=y.match(G_);_&&(n+=_.length);var k=y.lastIndexOf(iv);r=~k?y.length-k:r+y.length}function a(){var y={line:n,column:r};return function(_){return _.position=new o(y),u(),_}}function o(y){this.start=y,this.end={line:n,column:r},this.source=t.source}o.prototype.content=e;function s(y){var _=new Error(t.source+":"+n+":"+r+": "+y);if(_.reason=y,_.filename=t.source,_.line=n,_.column=r,_.source=e,!t.silent)throw _}function l(y){var _=y.exec(e);if(_){var k=_[0];return i(k),e=e.slice(k.length),_}}function u(){l(Z_)}function d(y){var _;for(y=y||[];_=c();)_!==!1&&y.push(_);return y}function c(){var y=a();if(!(op!=e.charAt(0)||sp!=e.charAt(1))){for(var _=2;qn!=e.charAt(_)&&(sp!=e.charAt(_)||op!=e.charAt(_+1));)++_;if(_+=2,qn===e.charAt(_-1))return s("End of comment missing");var k=e.slice(2,_-2);return r+=2,i(k),e=e.slice(_),r+=2,y({type:av,comment:k})}}function f(){var y=a(),_=l(J_);if(_){if(c(),!l(ev))return s("property missing ':'");var k=l(tv),h=y({type:ov,property:lp(_[0].replace(ap,qn)),value:k?lp(k[0].replace(ap,qn)):qn});return l(nv),h}}function m(){var y=[];d(y);for(var _;_=f();)_!==!1&&(y.push(_),d(y));return y}return u(),m()}function lp(e){return e?e.replace(rv,qn):qn}var lv=sv,uv=Xa&&Xa.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(pc,"__esModule",{value:!0});pc.default=dv;const cv=uv(lv);function dv(e,t){let n=null;if(!e||typeof e!="string")return n;const r=(0,cv.default)(e),i=typeof t=="function";return r.forEach(a=>{if(a.type!=="declaration")return;const{property:o,value:s}=a;i?t(o,s,a):s&&(n=n||{},n[o]=s)}),n}var Ko={};Object.defineProperty(Ko,"__esModule",{value:!0});Ko.camelCase=void 0;var pv=/^--[a-zA-Z0-9_-]+$/,fv=/-([a-z])/g,mv=/^[^-]+$/,hv=/^-(webkit|moz|ms|o|khtml)-/,gv=/^-(ms)-/,yv=function(e){return!e||mv.test(e)||pv.test(e)},xv=function(e,t){return t.toUpperCase()},up=function(e,t){return"".concat(t,"-")},_v=function(e,t){return t===void 0&&(t={}),yv(e)?e:(e=e.toLowerCase(),t.reactCompat?e=e.replace(gv,up):e=e.replace(hv,up),e.replace(fv,xv))};Ko.camelCase=_v;var vv=Xa&&Xa.__importDefault||function(e){return e&&e.__esModule?e:{default:e}},bv=vv(pc),wv=Ko;function ru(e,t){var n={};return!e||typeof e!="string"||(0,bv.default)(e,function(r,i){r&&i&&(n[(0,wv.camelCase)(r,t)]=i)}),n}ru.default=ru;var kv=ru;const Sv=Jn(kv),Kh=Yh("end"),fc=Yh("start");function Yh(e){return t;function t(n){const r=n&&n.position&&n.position[e]||{};if(typeof r.line=="number"&&r.line>0&&typeof r.column=="number"&&r.column>0)return{line:r.line,column:r.column,offset:typeof r.offset=="number"&&r.offset>-1?r.offset:void 0}}}function Cv(e){const t=fc(e),n=Kh(e);if(t&&n)return{start:t,end:n}}function ki(e){return!e||typeof e!="object"?"":"position"in e||"type"in e?cp(e.position):"start"in e||"end"in e?cp(e):"line"in e||"column"in e?iu(e):""}function iu(e){return dp(e&&e.line)+":"+dp(e&&e.column)}function cp(e){return iu(e&&e.start)+"-"+iu(e&&e.end)}function dp(e){return e&&typeof e=="number"?e:1}class We extends Error{constructor(t,n,r){super(),typeof n=="string"&&(r=n,n=void 0);let i="",a={},o=!1;if(n&&("line"in n&&"column"in n?a={place:n}:"start"in n&&"end"in n?a={place:n}:"type"in n?a={ancestors:[n],place:n.position}:a={...n}),typeof t=="string"?i=t:!a.cause&&t&&(o=!0,i=t.message,a.cause=t),!a.ruleId&&!a.source&&typeof r=="string"){const l=r.indexOf(":");l===-1?a.ruleId=r:(a.source=r.slice(0,l),a.ruleId=r.slice(l+1))}if(!a.place&&a.ancestors&&a.ancestors){const l=a.ancestors[a.ancestors.length-1];l&&(a.place=l.position)}const s=a.place&&"start"in a.place?a.place.start:a.place;this.ancestors=a.ancestors||void 0,this.cause=a.cause||void 0,this.column=s?s.column:void 0,this.fatal=void 0,this.file="",this.message=i,this.line=s?s.line:void 0,this.name=ki(a.place)||"1:1",this.place=a.place||void 0,this.reason=this.message,this.ruleId=a.ruleId||void 0,this.source=a.source||void 0,this.stack=o&&a.cause&&typeof a.cause.stack=="string"?a.cause.stack:"",this.actual=void 0,this.expected=void 0,this.note=void 0,this.url=void 0}}We.prototype.file="";We.prototype.name="";We.prototype.reason="";We.prototype.message="";We.prototype.stack="";We.prototype.column=void 0;We.prototype.line=void 0;We.prototype.ancestors=void 0;We.prototype.cause=void 0;We.prototype.fatal=void 0;We.prototype.place=void 0;We.prototype.ruleId=void 0;We.prototype.source=void 0;const mc={}.hasOwnProperty,Ev=new Map,zv=/[A-Z]/g,Tv=new Set(["table","tbody","thead","tfoot","tr"]),Pv=new Set(["td","th"]),Xh="https://github.com/syntax-tree/hast-util-to-jsx-runtime";function Nv(e,t){if(!t||t.Fragment===void 0)throw new TypeError("Expected `Fragment` in options");const n=t.filePath||void 0;let r;if(t.development){if(typeof t.jsxDEV!="function")throw new TypeError("Expected `jsxDEV` in options when `development: true`");r=Fv(n,t.jsxDEV)}else{if(typeof t.jsx!="function")throw new TypeError("Expected `jsx` in production options");if(typeof t.jsxs!="function")throw new TypeError("Expected `jsxs` in production options");r=Dv(n,t.jsx,t.jsxs)}const i={Fragment:t.Fragment,ancestors:[],components:t.components||{},create:r,elementAttributeNameCase:t.elementAttributeNameCase||"react",evaluater:t.createEvaluater?t.createEvaluater():void 0,filePath:n,ignoreInvalidStyle:t.ignoreInvalidStyle||!1,passKeys:t.passKeys!==!1,passNode:t.passNode||!1,schema:t.space==="svg"?dc:Y_,stylePropertyNameCase:t.stylePropertyNameCase||"dom",tableCellAlignToStyle:t.tableCellAlignToStyle!==!1},a=Gh(i,e,void 0);return a&&typeof a!="string"?a:i.create(e,i.Fragment,{children:a||void 0},void 0)}function Gh(e,t,n){if(t.type==="element")return jv(e,t,n);if(t.type==="mdxFlowExpression"||t.type==="mdxTextExpression")return Lv(e,t);if(t.type==="mdxJsxFlowElement"||t.type==="mdxJsxTextElement")return Rv(e,t,n);if(t.type==="mdxjsEsm")return Av(e,t);if(t.type==="root")return Iv(e,t,n);if(t.type==="text")return Ov(e,t)}function jv(e,t,n){const r=e.schema;let i=r;t.tagName.toLowerCase()==="svg"&&r.space==="html"&&(i=dc,e.schema=i),e.ancestors.push(t);const a=Jh(e,t.tagName,!1),o=Mv(e,t);let s=gc(e,t);return Tv.has(t.tagName)&&(s=s.filter(function(l){return typeof l=="string"?!M_(l):!0})),Zh(e,o,a,t),hc(o,s),e.ancestors.pop(),e.schema=r,e.create(t,a,o,n)}function Lv(e,t){if(t.data&&t.data.estree&&e.evaluater){const r=t.data.estree.body[0];return r.type,e.evaluater.evaluateExpression(r.expression)}$i(e,t.position)}function Av(e,t){if(t.data&&t.data.estree&&e.evaluater)return e.evaluater.evaluateProgram(t.data.estree);$i(e,t.position)}function Rv(e,t,n){const r=e.schema;let i=r;t.name==="svg"&&r.space==="html"&&(i=dc,e.schema=i),e.ancestors.push(t);const a=t.name===null?e.Fragment:Jh(e,t.name,!0),o=qv(e,t),s=gc(e,t);return Zh(e,o,a,t),hc(o,s),e.ancestors.pop(),e.schema=r,e.create(t,a,o,n)}function Iv(e,t,n){const r={};return hc(r,gc(e,t)),e.create(t,e.Fragment,r,n)}function Ov(e,t){return t.value}function Zh(e,t,n,r){typeof n!="string"&&n!==e.Fragment&&e.passNode&&(t.node=r)}function hc(e,t){if(t.length>0){const n=t.length>1?t:t[0];n&&(e.children=n)}}function Dv(e,t,n){return r;function r(i,a,o,s){const u=Array.isArray(o.children)?n:t;return s?u(a,o,s):u(a,o)}}function Fv(e,t){return n;function n(r,i,a,o){const s=Array.isArray(a.children),l=fc(r);return t(i,a,o,s,{columnNumber:l?l.column-1:void 0,fileName:e,lineNumber:l?l.line:void 0},void 0)}}function Mv(e,t){const n={};let r,i;for(i in t.properties)if(i!=="children"&&mc.call(t.properties,i)){const a=Wv(e,i,t.properties[i]);if(a){const[o,s]=a;e.tableCellAlignToStyle&&o==="align"&&typeof s=="string"&&Pv.has(t.tagName)?r=s:n[o]=s}}if(r){const a=n.style||(n.style={});a[e.stylePropertyNameCase==="css"?"text-align":"textAlign"]=r}return n}function qv(e,t){const n={};for(const r of t.attributes)if(r.type==="mdxJsxExpressionAttribute")if(r.data&&r.data.estree&&e.evaluater){const a=r.data.estree.body[0];a.type;const o=a.expression;o.type;const s=o.properties[0];s.type,Object.assign(n,e.evaluater.evaluateExpression(s.argument))}else $i(e,t.position);else{const i=r.name;let a;if(r.value&&typeof r.value=="object")if(r.value.data&&r.value.data.estree&&e.evaluater){const s=r.value.data.estree.body[0];s.type,a=e.evaluater.evaluateExpression(s.expression)}else $i(e,t.position);else a=r.value===null?!0:r.value;n[i]=a}return n}function gc(e,t){const n=[];let r=-1;const i=e.passKeys?new Map:Ev;for(;++r<t.children.length;){const a=t.children[r];let o;if(e.passKeys){const l=a.type==="element"?a.tagName:a.type==="mdxJsxFlowElement"||a.type==="mdxJsxTextElement"?a.name:void 0;if(l){const u=i.get(l)||0;o=l+"-"+u,i.set(l,u+1)}}const s=Gh(e,a,o);s!==void 0&&n.push(s)}return n}function Wv(e,t,n){const r=$_(e.schema,t);if(!(n==null||typeof n=="number"&&Number.isNaN(n))){if(Array.isArray(n)&&(n=r.commaSeparated?R_(n):X_(n)),r.property==="style"){let i=typeof n=="object"?n:Bv(e,String(n));return e.stylePropertyNameCase==="css"&&(i=Vv(i)),["style",i]}return[e.elementAttributeNameCase==="react"&&r.space?V_[r.property]||r.property:r.attribute,n]}}function Bv(e,t){try{return Sv(t,{reactCompat:!0})}catch(n){if(e.ignoreInvalidStyle)return{};const r=n,i=new We("Cannot parse `style` attribute",{ancestors:e.ancestors,cause:r,ruleId:"style",source:"hast-util-to-jsx-runtime"});throw i.file=e.filePath||void 0,i.url=Xh+"#cannot-parse-style-attribute",i}}function Jh(e,t,n){let r;if(!n)r={type:"Literal",value:t};else if(t.includes(".")){const i=t.split(".");let a=-1,o;for(;++a<i.length;){const s=tp(i[a])?{type:"Identifier",name:i[a]}:{type:"Literal",value:i[a]};o=o?{type:"MemberExpression",object:o,property:s,computed:!!(a&&s.type==="Literal"),optional:!1}:s}r=o}else r=tp(t)&&!/^[a-z]/.test(t)?{type:"Identifier",name:t}:{type:"Literal",value:t};if(r.type==="Literal"){const i=r.value;return mc.call(e.components,i)?e.components[i]:i}if(e.evaluater)return e.evaluater.evaluateExpression(r);$i(e)}function $i(e,t){const n=new We("Cannot handle MDX estrees without `createEvaluater`",{ancestors:e.ancestors,place:t,ruleId:"mdx-estree",source:"hast-util-to-jsx-runtime"});throw n.file=e.filePath||void 0,n.url=Xh+"#cannot-handle-mdx-estrees-without-createevaluater",n}function Vv(e){const t={};let n;for(n in e)mc.call(e,n)&&(t[Uv(n)]=e[n]);return t}function Uv(e){let t=e.replace(zv,Hv);return t.slice(0,3)==="ms-"&&(t="-"+t),t}function Hv(e){return"-"+e.toLowerCase()}const As={action:["form"],cite:["blockquote","del","ins","q"],data:["object"],formAction:["button","input"],href:["a","area","base","link"],icon:["menuitem"],itemId:null,manifest:["html"],ping:["a","area"],poster:["video"],src:["audio","embed","iframe","img","input","script","source","track","video"]},$v={};function yc(e,t){const n=$v,r=typeof n.includeImageAlt=="boolean"?n.includeImageAlt:!0,i=typeof n.includeHtml=="boolean"?n.includeHtml:!0;return eg(e,r,i)}function eg(e,t,n){if(Qv(e)){if("value"in e)return e.type==="html"&&!n?"":e.value;if(t&&"alt"in e&&e.alt)return e.alt;if("children"in e)return pp(e.children,t,n)}return Array.isArray(e)?pp(e,t,n):""}function pp(e,t,n){const r=[];let i=-1;for(;++i<e.length;)r[i]=eg(e[i],t,n);return r.join("")}function Qv(e){return!!(e&&typeof e=="object")}const fp=document.createElement("i");function xc(e){const t="&"+e+";";fp.innerHTML=t;const n=fp.textContent;return n.charCodeAt(n.length-1)===59&&e!=="semi"||n===t?!1:n}function lt(e,t,n,r){const i=e.length;let a=0,o;if(t<0?t=-t>i?0:i+t:t=t>i?i:t,n=n>0?n:0,r.length<1e4)o=Array.from(r),o.unshift(t,n),e.splice(...o);else for(n&&e.splice(t,n);a<r.length;)o=r.slice(a,a+1e4),o.unshift(t,0),e.splice(...o),a+=1e4,t+=1e4}function yt(e,t){return e.length>0?(lt(e,e.length,0,t),e):t}const mp={}.hasOwnProperty;function tg(e){const t={};let n=-1;for(;++n<e.length;)Kv(t,e[n]);return t}function Kv(e,t){let n;for(n in t){const i=(mp.call(e,n)?e[n]:void 0)||(e[n]={}),a=t[n];let o;if(a)for(o in a){mp.call(i,o)||(i[o]=[]);const s=a[o];Yv(i[o],Array.isArray(s)?s:s?[s]:[])}}}function Yv(e,t){let n=-1;const r=[];for(;++n<t.length;)(t[n].add==="after"?e:r).push(t[n]);lt(e,0,0,r)}function ng(e,t){const n=Number.parseInt(e,t);return n<9||n===11||n>13&&n<32||n>126&&n<160||n>55295&&n<57344||n>64975&&n<65008||(n&65535)===65535||(n&65535)===65534||n>1114111?"�":String.fromCodePoint(n)}function Rt(e){return e.replace(/[\t\n\r ]+/g," ").replace(/^ | $/g,"").toLowerCase().toUpperCase()}const Ue=Ln(/[A-Za-z]/),Me=Ln(/[\dA-Za-z]/),Xv=Ln(/[#-'*+\--9=?A-Z^-~]/);function So(e){return e!==null&&(e<32||e===127)}const au=Ln(/\d/),Gv=Ln(/[\dA-Fa-f]/),Zv=Ln(/[!-/:-@[-`{-~]/);function K(e){return e!==null&&e<-2}function ce(e){return e!==null&&(e<0||e===32)}function Z(e){return e===-2||e===-1||e===32}const Yo=Ln(new RegExp("\\p{P}|\\p{S}","u")),Zn=Ln(/\s/);function Ln(e){return t;function t(n){return n!==null&&n>-1&&e.test(String.fromCharCode(n))}}function Qr(e){const t=[];let n=-1,r=0,i=0;for(;++n<e.length;){const a=e.charCodeAt(n);let o="";if(a===37&&Me(e.charCodeAt(n+1))&&Me(e.charCodeAt(n+2)))i=2;else if(a<128)/[!#$&-;=?-Z_a-z~]/.test(String.fromCharCode(a))||(o=String.fromCharCode(a));else if(a>55295&&a<57344){const s=e.charCodeAt(n+1);a<56320&&s>56319&&s<57344?(o=String.fromCharCode(a,s),i=1):o="�"}else o=String.fromCharCode(a);o&&(t.push(e.slice(r,n),encodeURIComponent(o)),r=n+i+1,o=""),i&&(n+=i,i=0)}return t.join("")+e.slice(r)}function re(e,t,n,r){const i=r?r-1:Number.POSITIVE_INFINITY;let a=0;return o;function o(l){return Z(l)?(e.enter(n),s(l)):t(l)}function s(l){return Z(l)&&a++<i?(e.consume(l),s):(e.exit(n),t(l))}}const Jv={tokenize:eb};function eb(e){const t=e.attempt(this.parser.constructs.contentInitial,r,i);let n;return t;function r(s){if(s===null){e.consume(s);return}return e.enter("lineEnding"),e.consume(s),e.exit("lineEnding"),re(e,t,"linePrefix")}function i(s){return e.enter("paragraph"),a(s)}function a(s){const l=e.enter("chunkText",{contentType:"text",previous:n});return n&&(n.next=l),n=l,o(s)}function o(s){if(s===null){e.exit("chunkText"),e.exit("paragraph"),e.consume(s);return}return K(s)?(e.consume(s),e.exit("chunkText"),a):(e.consume(s),o)}}const tb={tokenize:nb},hp={tokenize:rb};function nb(e){const t=this,n=[];let r=0,i,a,o;return s;function s(x){if(r<n.length){const C=n[r];return t.containerState=C[1],e.attempt(C[0].continuation,l,u)(x)}return u(x)}function l(x){if(r++,t.containerState._closeFlow){t.containerState._closeFlow=void 0,i&&g();const C=t.events.length;let z=C,S;for(;z--;)if(t.events[z][0]==="exit"&&t.events[z][1].type==="chunkFlow"){S=t.events[z][1].end;break}h(r);let T=C;for(;T<t.events.length;)t.events[T][1].end={...S},T++;return lt(t.events,z+1,0,t.events.slice(C)),t.events.length=T,u(x)}return s(x)}function u(x){if(r===n.length){if(!i)return f(x);if(i.currentConstruct&&i.currentConstruct.concrete)return y(x);t.interrupt=!!(i.currentConstruct&&!i._gfmTableDynamicInterruptHack)}return t.containerState={},e.check(hp,d,c)(x)}function d(x){return i&&g(),h(r),f(x)}function c(x){return t.parser.lazy[t.now().line]=r!==n.length,o=t.now().offset,y(x)}function f(x){return t.containerState={},e.attempt(hp,m,y)(x)}function m(x){return r++,n.push([t.currentConstruct,t.containerState]),f(x)}function y(x){if(x===null){i&&g(),h(0),e.consume(x);return}return i=i||t.parser.flow(t.now()),e.enter("chunkFlow",{_tokenizer:i,contentType:"flow",previous:a}),_(x)}function _(x){if(x===null){k(e.exit("chunkFlow"),!0),h(0),e.consume(x);return}return K(x)?(e.consume(x),k(e.exit("chunkFlow")),r=0,t.interrupt=void 0,s):(e.consume(x),_)}function k(x,C){const z=t.sliceStream(x);if(C&&z.push(null),x.previous=a,a&&(a.next=x),a=x,i.defineSkip(x.start),i.write(z),t.parser.lazy[x.start.line]){let S=i.events.length;for(;S--;)if(i.events[S][1].start.offset<o&&(!i.events[S][1].end||i.events[S][1].end.offset>o))return;const T=t.events.length;let N=T,M,w;for(;N--;)if(t.events[N][0]==="exit"&&t.events[N][1].type==="chunkFlow"){if(M){w=t.events[N][1].end;break}M=!0}for(h(r),S=T;S<t.events.length;)t.events[S][1].end={...w},S++;lt(t.events,N+1,0,t.events.slice(T)),t.events.length=S}}function h(x){let C=n.length;for(;C-- >x;){const z=n[C];t.containerState=z[1],z[0].exit.call(t,e)}n.length=x}function g(){i.write([null]),a=void 0,i=void 0,t.containerState._closeFlow=void 0}}function rb(e,t,n){return re(e,e.attempt(this.parser.constructs.document,t,n),"linePrefix",this.parser.constructs.disable.null.includes("codeIndented")?void 0:4)}function qr(e){if(e===null||ce(e)||Zn(e))return 1;if(Yo(e))return 2}function Xo(e,t,n){const r=[];let i=-1;for(;++i<e.length;){const a=e[i].resolveAll;a&&!r.includes(a)&&(t=a(t,n),r.push(a))}return t}const ou={name:"attention",resolveAll:ib,tokenize:ab};function ib(e,t){let n=-1,r,i,a,o,s,l,u,d;for(;++n<e.length;)if(e[n][0]==="enter"&&e[n][1].type==="attentionSequence"&&e[n][1]._close){for(r=n;r--;)if(e[r][0]==="exit"&&e[r][1].type==="attentionSequence"&&e[r][1]._open&&t.sliceSerialize(e[r][1]).charCodeAt(0)===t.sliceSerialize(e[n][1]).charCodeAt(0)){if((e[r][1]._close||e[n][1]._open)&&(e[n][1].end.offset-e[n][1].start.offset)%3&&!((e[r][1].end.offset-e[r][1].start.offset+e[n][1].end.offset-e[n][1].start.offset)%3))continue;l=e[r][1].end.offset-e[r][1].start.offset>1&&e[n][1].end.offset-e[n][1].start.offset>1?2:1;const c={...e[r][1].end},f={...e[n][1].start};gp(c,-l),gp(f,l),o={type:l>1?"strongSequence":"emphasisSequence",start:c,end:{...e[r][1].end}},s={type:l>1?"strongSequence":"emphasisSequence",start:{...e[n][1].start},end:f},a={type:l>1?"strongText":"emphasisText",start:{...e[r][1].end},end:{...e[n][1].start}},i={type:l>1?"strong":"emphasis",start:{...o.start},end:{...s.end}},e[r][1].end={...o.start},e[n][1].start={...s.end},u=[],e[r][1].end.offset-e[r][1].start.offset&&(u=yt(u,[["enter",e[r][1],t],["exit",e[r][1],t]])),u=yt(u,[["enter",i,t],["enter",o,t],["exit",o,t],["enter",a,t]]),u=yt(u,Xo(t.parser.constructs.insideSpan.null,e.slice(r+1,n),t)),u=yt(u,[["exit",a,t],["enter",s,t],["exit",s,t],["exit",i,t]]),e[n][1].end.offset-e[n][1].start.offset?(d=2,u=yt(u,[["enter",e[n][1],t],["exit",e[n][1],t]])):d=0,lt(e,r-1,n-r+3,u),n=r+u.length-d-2;break}}for(n=-1;++n<e.length;)e[n][1].type==="attentionSequence"&&(e[n][1].type="data");return e}function ab(e,t){const n=this.parser.constructs.attentionMarkers.null,r=this.previous,i=qr(r);let a;return o;function o(l){return a=l,e.enter("attentionSequence"),s(l)}function s(l){if(l===a)return e.consume(l),s;const u=e.exit("attentionSequence"),d=qr(l),c=!d||d===2&&i||n.includes(l),f=!i||i===2&&d||n.includes(r);return u._open=!!(a===42?c:c&&(i||!f)),u._close=!!(a===42?f:f&&(d||!c)),t(l)}}function gp(e,t){e.column+=t,e.offset+=t,e._bufferIndex+=t}const ob={name:"autolink",tokenize:sb};function sb(e,t,n){let r=0;return i;function i(m){return e.enter("autolink"),e.enter("autolinkMarker"),e.consume(m),e.exit("autolinkMarker"),e.enter("autolinkProtocol"),a}function a(m){return Ue(m)?(e.consume(m),o):m===64?n(m):u(m)}function o(m){return m===43||m===45||m===46||Me(m)?(r=1,s(m)):u(m)}function s(m){return m===58?(e.consume(m),r=0,l):(m===43||m===45||m===46||Me(m))&&r++<32?(e.consume(m),s):(r=0,u(m))}function l(m){return m===62?(e.exit("autolinkProtocol"),e.enter("autolinkMarker"),e.consume(m),e.exit("autolinkMarker"),e.exit("autolink"),t):m===null||m===32||m===60||So(m)?n(m):(e.consume(m),l)}function u(m){return m===64?(e.consume(m),d):Xv(m)?(e.consume(m),u):n(m)}function d(m){return Me(m)?c(m):n(m)}function c(m){return m===46?(e.consume(m),r=0,d):m===62?(e.exit("autolinkProtocol").type="autolinkEmail",e.enter("autolinkMarker"),e.consume(m),e.exit("autolinkMarker"),e.exit("autolink"),t):f(m)}function f(m){if((m===45||Me(m))&&r++<63){const y=m===45?f:c;return e.consume(m),y}return n(m)}}const aa={partial:!0,tokenize:lb};function lb(e,t,n){return r;function r(a){return Z(a)?re(e,i,"linePrefix")(a):i(a)}function i(a){return a===null||K(a)?t(a):n(a)}}const rg={continuation:{tokenize:cb},exit:db,name:"blockQuote",tokenize:ub};function ub(e,t,n){const r=this;return i;function i(o){if(o===62){const s=r.containerState;return s.open||(e.enter("blockQuote",{_container:!0}),s.open=!0),e.enter("blockQuotePrefix"),e.enter("blockQuoteMarker"),e.consume(o),e.exit("blockQuoteMarker"),a}return n(o)}function a(o){return Z(o)?(e.enter("blockQuotePrefixWhitespace"),e.consume(o),e.exit("blockQuotePrefixWhitespace"),e.exit("blockQuotePrefix"),t):(e.exit("blockQuotePrefix"),t(o))}}function cb(e,t,n){const r=this;return i;function i(o){return Z(o)?re(e,a,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(o):a(o)}function a(o){return e.attempt(rg,t,n)(o)}}function db(e){e.exit("blockQuote")}const ig={name:"characterEscape",tokenize:pb};function pb(e,t,n){return r;function r(a){return e.enter("characterEscape"),e.enter("escapeMarker"),e.consume(a),e.exit("escapeMarker"),i}function i(a){return Zv(a)?(e.enter("characterEscapeValue"),e.consume(a),e.exit("characterEscapeValue"),e.exit("characterEscape"),t):n(a)}}const ag={name:"characterReference",tokenize:fb};function fb(e,t,n){const r=this;let i=0,a,o;return s;function s(c){return e.enter("characterReference"),e.enter("characterReferenceMarker"),e.consume(c),e.exit("characterReferenceMarker"),l}function l(c){return c===35?(e.enter("characterReferenceMarkerNumeric"),e.consume(c),e.exit("characterReferenceMarkerNumeric"),u):(e.enter("characterReferenceValue"),a=31,o=Me,d(c))}function u(c){return c===88||c===120?(e.enter("characterReferenceMarkerHexadecimal"),e.consume(c),e.exit("characterReferenceMarkerHexadecimal"),e.enter("characterReferenceValue"),a=6,o=Gv,d):(e.enter("characterReferenceValue"),a=7,o=au,d(c))}function d(c){if(c===59&&i){const f=e.exit("characterReferenceValue");return o===Me&&!xc(r.sliceSerialize(f))?n(c):(e.enter("characterReferenceMarker"),e.consume(c),e.exit("characterReferenceMarker"),e.exit("characterReference"),t)}return o(c)&&i++<a?(e.consume(c),d):n(c)}}const yp={partial:!0,tokenize:hb},xp={concrete:!0,name:"codeFenced",tokenize:mb};function mb(e,t,n){const r=this,i={partial:!0,tokenize:z};let a=0,o=0,s;return l;function l(S){return u(S)}function u(S){const T=r.events[r.events.length-1];return a=T&&T[1].type==="linePrefix"?T[2].sliceSerialize(T[1],!0).length:0,s=S,e.enter("codeFenced"),e.enter("codeFencedFence"),e.enter("codeFencedFenceSequence"),d(S)}function d(S){return S===s?(o++,e.consume(S),d):o<3?n(S):(e.exit("codeFencedFenceSequence"),Z(S)?re(e,c,"whitespace")(S):c(S))}function c(S){return S===null||K(S)?(e.exit("codeFencedFence"),r.interrupt?t(S):e.check(yp,_,C)(S)):(e.enter("codeFencedFenceInfo"),e.enter("chunkString",{contentType:"string"}),f(S))}function f(S){return S===null||K(S)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),c(S)):Z(S)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),re(e,m,"whitespace")(S)):S===96&&S===s?n(S):(e.consume(S),f)}function m(S){return S===null||K(S)?c(S):(e.enter("codeFencedFenceMeta"),e.enter("chunkString",{contentType:"string"}),y(S))}function y(S){return S===null||K(S)?(e.exit("chunkString"),e.exit("codeFencedFenceMeta"),c(S)):S===96&&S===s?n(S):(e.consume(S),y)}function _(S){return e.attempt(i,C,k)(S)}function k(S){return e.enter("lineEnding"),e.consume(S),e.exit("lineEnding"),h}function h(S){return a>0&&Z(S)?re(e,g,"linePrefix",a+1)(S):g(S)}function g(S){return S===null||K(S)?e.check(yp,_,C)(S):(e.enter("codeFlowValue"),x(S))}function x(S){return S===null||K(S)?(e.exit("codeFlowValue"),g(S)):(e.consume(S),x)}function C(S){return e.exit("codeFenced"),t(S)}function z(S,T,N){let M=0;return w;function w(j){return S.enter("lineEnding"),S.consume(j),S.exit("lineEnding"),I}function I(j){return S.enter("codeFencedFence"),Z(j)?re(S,W,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(j):W(j)}function W(j){return j===s?(S.enter("codeFencedFenceSequence"),Y(j)):N(j)}function Y(j){return j===s?(M++,S.consume(j),Y):M>=o?(S.exit("codeFencedFenceSequence"),Z(j)?re(S,ee,"whitespace")(j):ee(j)):N(j)}function ee(j){return j===null||K(j)?(S.exit("codeFencedFence"),T(j)):N(j)}}}function hb(e,t,n){const r=this;return i;function i(o){return o===null?n(o):(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),a)}function a(o){return r.parser.lazy[r.now().line]?n(o):t(o)}}const Rs={name:"codeIndented",tokenize:yb},gb={partial:!0,tokenize:xb};function yb(e,t,n){const r=this;return i;function i(u){return e.enter("codeIndented"),re(e,a,"linePrefix",5)(u)}function a(u){const d=r.events[r.events.length-1];return d&&d[1].type==="linePrefix"&&d[2].sliceSerialize(d[1],!0).length>=4?o(u):n(u)}function o(u){return u===null?l(u):K(u)?e.attempt(gb,o,l)(u):(e.enter("codeFlowValue"),s(u))}function s(u){return u===null||K(u)?(e.exit("codeFlowValue"),o(u)):(e.consume(u),s)}function l(u){return e.exit("codeIndented"),t(u)}}function xb(e,t,n){const r=this;return i;function i(o){return r.parser.lazy[r.now().line]?n(o):K(o)?(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),i):re(e,a,"linePrefix",5)(o)}function a(o){const s=r.events[r.events.length-1];return s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(o):K(o)?i(o):n(o)}}const _b={name:"codeText",previous:bb,resolve:vb,tokenize:wb};function vb(e){let t=e.length-4,n=3,r,i;if((e[n][1].type==="lineEnding"||e[n][1].type==="space")&&(e[t][1].type==="lineEnding"||e[t][1].type==="space")){for(r=n;++r<t;)if(e[r][1].type==="codeTextData"){e[n][1].type="codeTextPadding",e[t][1].type="codeTextPadding",n+=2,t-=2;break}}for(r=n-1,t++;++r<=t;)i===void 0?r!==t&&e[r][1].type!=="lineEnding"&&(i=r):(r===t||e[r][1].type==="lineEnding")&&(e[i][1].type="codeTextData",r!==i+2&&(e[i][1].end=e[r-1][1].end,e.splice(i+2,r-i-2),t-=r-i-2,r=i+2),i=void 0);return e}function bb(e){return e!==96||this.events[this.events.length-1][1].type==="characterEscape"}function wb(e,t,n){let r=0,i,a;return o;function o(c){return e.enter("codeText"),e.enter("codeTextSequence"),s(c)}function s(c){return c===96?(e.consume(c),r++,s):(e.exit("codeTextSequence"),l(c))}function l(c){return c===null?n(c):c===32?(e.enter("space"),e.consume(c),e.exit("space"),l):c===96?(a=e.enter("codeTextSequence"),i=0,d(c)):K(c)?(e.enter("lineEnding"),e.consume(c),e.exit("lineEnding"),l):(e.enter("codeTextData"),u(c))}function u(c){return c===null||c===32||c===96||K(c)?(e.exit("codeTextData"),l(c)):(e.consume(c),u)}function d(c){return c===96?(e.consume(c),i++,d):i===r?(e.exit("codeTextSequence"),e.exit("codeText"),t(c)):(a.type="codeTextData",u(c))}}class kb{constructor(t){this.left=t?[...t]:[],this.right=[]}get(t){if(t<0||t>=this.left.length+this.right.length)throw new RangeError("Cannot access index `"+t+"` in a splice buffer of size `"+(this.left.length+this.right.length)+"`");return t<this.left.length?this.left[t]:this.right[this.right.length-t+this.left.length-1]}get length(){return this.left.length+this.right.length}shift(){return this.setCursor(0),this.right.pop()}slice(t,n){const r=n??Number.POSITIVE_INFINITY;return r<this.left.length?this.left.slice(t,r):t>this.left.length?this.right.slice(this.right.length-r+this.left.length,this.right.length-t+this.left.length).reverse():this.left.slice(t).concat(this.right.slice(this.right.length-r+this.left.length).reverse())}splice(t,n,r){const i=n||0;this.setCursor(Math.trunc(t));const a=this.right.splice(this.right.length-i,Number.POSITIVE_INFINITY);return r&&oi(this.left,r),a.reverse()}pop(){return this.setCursor(Number.POSITIVE_INFINITY),this.left.pop()}push(t){this.setCursor(Number.POSITIVE_INFINITY),this.left.push(t)}pushMany(t){this.setCursor(Number.POSITIVE_INFINITY),oi(this.left,t)}unshift(t){this.setCursor(0),this.right.push(t)}unshiftMany(t){this.setCursor(0),oi(this.right,t.reverse())}setCursor(t){if(!(t===this.left.length||t>this.left.length&&this.right.length===0||t<0&&this.left.length===0))if(t<this.left.length){const n=this.left.splice(t,Number.POSITIVE_INFINITY);oi(this.right,n.reverse())}else{const n=this.right.splice(this.left.length+this.right.length-t,Number.POSITIVE_INFINITY);oi(this.left,n.reverse())}}}function oi(e,t){let n=0;if(t.length<1e4)e.push(...t);else for(;n<t.length;)e.push(...t.slice(n,n+1e4)),n+=1e4}function og(e){const t={};let n=-1,r,i,a,o,s,l,u;const d=new kb(e);for(;++n<d.length;){for(;n in t;)n=t[n];if(r=d.get(n),n&&r[1].type==="chunkFlow"&&d.get(n-1)[1].type==="listItemPrefix"&&(l=r[1]._tokenizer.events,a=0,a<l.length&&l[a][1].type==="lineEndingBlank"&&(a+=2),a<l.length&&l[a][1].type==="content"))for(;++a<l.length&&l[a][1].type!=="content";)l[a][1].type==="chunkText"&&(l[a][1]._isInFirstContentOfListItem=!0,a++);if(r[0]==="enter")r[1].contentType&&(Object.assign(t,Sb(d,n)),n=t[n],u=!0);else if(r[1]._container){for(a=n,i=void 0;a--;)if(o=d.get(a),o[1].type==="lineEnding"||o[1].type==="lineEndingBlank")o[0]==="enter"&&(i&&(d.get(i)[1].type="lineEndingBlank"),o[1].type="lineEnding",i=a);else if(!(o[1].type==="linePrefix"||o[1].type==="listItemIndent"))break;i&&(r[1].end={...d.get(i)[1].start},s=d.slice(i,n),s.unshift(r),d.splice(i,n-i+1,s))}}return lt(e,0,Number.POSITIVE_INFINITY,d.slice(0)),!u}function Sb(e,t){const n=e.get(t)[1],r=e.get(t)[2];let i=t-1;const a=[];let o=n._tokenizer;o||(o=r.parser[n.contentType](n.start),n._contentTypeTextTrailing&&(o._contentTypeTextTrailing=!0));const s=o.events,l=[],u={};let d,c,f=-1,m=n,y=0,_=0;const k=[_];for(;m;){for(;e.get(++i)[1]!==m;);a.push(i),m._tokenizer||(d=r.sliceStream(m),m.next||d.push(null),c&&o.defineSkip(m.start),m._isInFirstContentOfListItem&&(o._gfmTasklistFirstContentOfListItem=!0),o.write(d),m._isInFirstContentOfListItem&&(o._gfmTasklistFirstContentOfListItem=void 0)),c=m,m=m.next}for(m=n;++f<s.length;)s[f][0]==="exit"&&s[f-1][0]==="enter"&&s[f][1].type===s[f-1][1].type&&s[f][1].start.line!==s[f][1].end.line&&(_=f+1,k.push(_),m._tokenizer=void 0,m.previous=void 0,m=m.next);for(o.events=[],m?(m._tokenizer=void 0,m.previous=void 0):k.pop(),f=k.length;f--;){const h=s.slice(k[f],k[f+1]),g=a.pop();l.push([g,g+h.length-1]),e.splice(g,2,h)}for(l.reverse(),f=-1;++f<l.length;)u[y+l[f][0]]=y+l[f][1],y+=l[f][1]-l[f][0]-1;return u}const Cb={resolve:zb,tokenize:Tb},Eb={partial:!0,tokenize:Pb};function zb(e){return og(e),e}function Tb(e,t){let n;return r;function r(s){return e.enter("content"),n=e.enter("chunkContent",{contentType:"content"}),i(s)}function i(s){return s===null?a(s):K(s)?e.check(Eb,o,a)(s):(e.consume(s),i)}function a(s){return e.exit("chunkContent"),e.exit("content"),t(s)}function o(s){return e.consume(s),e.exit("chunkContent"),n.next=e.enter("chunkContent",{contentType:"content",previous:n}),n=n.next,i}}function Pb(e,t,n){const r=this;return i;function i(o){return e.exit("chunkContent"),e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),re(e,a,"linePrefix")}function a(o){if(o===null||K(o))return n(o);const s=r.events[r.events.length-1];return!r.parser.constructs.disable.null.includes("codeIndented")&&s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(o):e.interrupt(r.parser.constructs.flow,n,t)(o)}}function sg(e,t,n,r,i,a,o,s,l){const u=l||Number.POSITIVE_INFINITY;let d=0;return c;function c(h){return h===60?(e.enter(r),e.enter(i),e.enter(a),e.consume(h),e.exit(a),f):h===null||h===32||h===41||So(h)?n(h):(e.enter(r),e.enter(o),e.enter(s),e.enter("chunkString",{contentType:"string"}),_(h))}function f(h){return h===62?(e.enter(a),e.consume(h),e.exit(a),e.exit(i),e.exit(r),t):(e.enter(s),e.enter("chunkString",{contentType:"string"}),m(h))}function m(h){return h===62?(e.exit("chunkString"),e.exit(s),f(h)):h===null||h===60||K(h)?n(h):(e.consume(h),h===92?y:m)}function y(h){return h===60||h===62||h===92?(e.consume(h),m):m(h)}function _(h){return!d&&(h===null||h===41||ce(h))?(e.exit("chunkString"),e.exit(s),e.exit(o),e.exit(r),t(h)):d<u&&h===40?(e.consume(h),d++,_):h===41?(e.consume(h),d--,_):h===null||h===32||h===40||So(h)?n(h):(e.consume(h),h===92?k:_)}function k(h){return h===40||h===41||h===92?(e.consume(h),_):_(h)}}function lg(e,t,n,r,i,a){const o=this;let s=0,l;return u;function u(m){return e.enter(r),e.enter(i),e.consume(m),e.exit(i),e.enter(a),d}function d(m){return s>999||m===null||m===91||m===93&&!l||m===94&&!s&&"_hiddenFootnoteSupport"in o.parser.constructs?n(m):m===93?(e.exit(a),e.enter(i),e.consume(m),e.exit(i),e.exit(r),t):K(m)?(e.enter("lineEnding"),e.consume(m),e.exit("lineEnding"),d):(e.enter("chunkString",{contentType:"string"}),c(m))}function c(m){return m===null||m===91||m===93||K(m)||s++>999?(e.exit("chunkString"),d(m)):(e.consume(m),l||(l=!Z(m)),m===92?f:c)}function f(m){return m===91||m===92||m===93?(e.consume(m),s++,c):c(m)}}function ug(e,t,n,r,i,a){let o;return s;function s(f){return f===34||f===39||f===40?(e.enter(r),e.enter(i),e.consume(f),e.exit(i),o=f===40?41:f,l):n(f)}function l(f){return f===o?(e.enter(i),e.consume(f),e.exit(i),e.exit(r),t):(e.enter(a),u(f))}function u(f){return f===o?(e.exit(a),l(o)):f===null?n(f):K(f)?(e.enter("lineEnding"),e.consume(f),e.exit("lineEnding"),re(e,u,"linePrefix")):(e.enter("chunkString",{contentType:"string"}),d(f))}function d(f){return f===o||f===null||K(f)?(e.exit("chunkString"),u(f)):(e.consume(f),f===92?c:d)}function c(f){return f===o||f===92?(e.consume(f),d):d(f)}}function Si(e,t){let n;return r;function r(i){return K(i)?(e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),n=!0,r):Z(i)?re(e,r,n?"linePrefix":"lineSuffix")(i):t(i)}}const Nb={name:"definition",tokenize:Lb},jb={partial:!0,tokenize:Ab};function Lb(e,t,n){const r=this;let i;return a;function a(m){return e.enter("definition"),o(m)}function o(m){return lg.call(r,e,s,n,"definitionLabel","definitionLabelMarker","definitionLabelString")(m)}function s(m){return i=Rt(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)),m===58?(e.enter("definitionMarker"),e.consume(m),e.exit("definitionMarker"),l):n(m)}function l(m){return ce(m)?Si(e,u)(m):u(m)}function u(m){return sg(e,d,n,"definitionDestination","definitionDestinationLiteral","definitionDestinationLiteralMarker","definitionDestinationRaw","definitionDestinationString")(m)}function d(m){return e.attempt(jb,c,c)(m)}function c(m){return Z(m)?re(e,f,"whitespace")(m):f(m)}function f(m){return m===null||K(m)?(e.exit("definition"),r.parser.defined.push(i),t(m)):n(m)}}function Ab(e,t,n){return r;function r(s){return ce(s)?Si(e,i)(s):n(s)}function i(s){return ug(e,a,n,"definitionTitle","definitionTitleMarker","definitionTitleString")(s)}function a(s){return Z(s)?re(e,o,"whitespace")(s):o(s)}function o(s){return s===null||K(s)?t(s):n(s)}}const Rb={name:"hardBreakEscape",tokenize:Ib};function Ib(e,t,n){return r;function r(a){return e.enter("hardBreakEscape"),e.consume(a),i}function i(a){return K(a)?(e.exit("hardBreakEscape"),t(a)):n(a)}}const Ob={name:"headingAtx",resolve:Db,tokenize:Fb};function Db(e,t){let n=e.length-2,r=3,i,a;return e[r][1].type==="whitespace"&&(r+=2),n-2>r&&e[n][1].type==="whitespace"&&(n-=2),e[n][1].type==="atxHeadingSequence"&&(r===n-1||n-4>r&&e[n-2][1].type==="whitespace")&&(n-=r+1===n?2:4),n>r&&(i={type:"atxHeadingText",start:e[r][1].start,end:e[n][1].end},a={type:"chunkText",start:e[r][1].start,end:e[n][1].end,contentType:"text"},lt(e,r,n-r+1,[["enter",i,t],["enter",a,t],["exit",a,t],["exit",i,t]])),e}function Fb(e,t,n){let r=0;return i;function i(d){return e.enter("atxHeading"),a(d)}function a(d){return e.enter("atxHeadingSequence"),o(d)}function o(d){return d===35&&r++<6?(e.consume(d),o):d===null||ce(d)?(e.exit("atxHeadingSequence"),s(d)):n(d)}function s(d){return d===35?(e.enter("atxHeadingSequence"),l(d)):d===null||K(d)?(e.exit("atxHeading"),t(d)):Z(d)?re(e,s,"whitespace")(d):(e.enter("atxHeadingText"),u(d))}function l(d){return d===35?(e.consume(d),l):(e.exit("atxHeadingSequence"),s(d))}function u(d){return d===null||d===35||ce(d)?(e.exit("atxHeadingText"),s(d)):(e.consume(d),u)}}const Mb=["address","article","aside","base","basefont","blockquote","body","caption","center","col","colgroup","dd","details","dialog","dir","div","dl","dt","fieldset","figcaption","figure","footer","form","frame","frameset","h1","h2","h3","h4","h5","h6","head","header","hr","html","iframe","legend","li","link","main","menu","menuitem","nav","noframes","ol","optgroup","option","p","param","search","section","summary","table","tbody","td","tfoot","th","thead","title","tr","track","ul"],_p=["pre","script","style","textarea"],qb={concrete:!0,name:"htmlFlow",resolveTo:Vb,tokenize:Ub},Wb={partial:!0,tokenize:$b},Bb={partial:!0,tokenize:Hb};function Vb(e){let t=e.length;for(;t--&&!(e[t][0]==="enter"&&e[t][1].type==="htmlFlow"););return t>1&&e[t-2][1].type==="linePrefix"&&(e[t][1].start=e[t-2][1].start,e[t+1][1].start=e[t-2][1].start,e.splice(t-2,2)),e}function Ub(e,t,n){const r=this;let i,a,o,s,l;return u;function u(b){return d(b)}function d(b){return e.enter("htmlFlow"),e.enter("htmlFlowData"),e.consume(b),c}function c(b){return b===33?(e.consume(b),f):b===47?(e.consume(b),a=!0,_):b===63?(e.consume(b),i=3,r.interrupt?t:v):Ue(b)?(e.consume(b),o=String.fromCharCode(b),k):n(b)}function f(b){return b===45?(e.consume(b),i=2,m):b===91?(e.consume(b),i=5,s=0,y):Ue(b)?(e.consume(b),i=4,r.interrupt?t:v):n(b)}function m(b){return b===45?(e.consume(b),r.interrupt?t:v):n(b)}function y(b){const ie="CDATA[";return b===ie.charCodeAt(s++)?(e.consume(b),s===ie.length?r.interrupt?t:W:y):n(b)}function _(b){return Ue(b)?(e.consume(b),o=String.fromCharCode(b),k):n(b)}function k(b){if(b===null||b===47||b===62||ce(b)){const ie=b===47,we=o.toLowerCase();return!ie&&!a&&_p.includes(we)?(i=1,r.interrupt?t(b):W(b)):Mb.includes(o.toLowerCase())?(i=6,ie?(e.consume(b),h):r.interrupt?t(b):W(b)):(i=7,r.interrupt&&!r.parser.lazy[r.now().line]?n(b):a?g(b):x(b))}return b===45||Me(b)?(e.consume(b),o+=String.fromCharCode(b),k):n(b)}function h(b){return b===62?(e.consume(b),r.interrupt?t:W):n(b)}function g(b){return Z(b)?(e.consume(b),g):w(b)}function x(b){return b===47?(e.consume(b),w):b===58||b===95||Ue(b)?(e.consume(b),C):Z(b)?(e.consume(b),x):w(b)}function C(b){return b===45||b===46||b===58||b===95||Me(b)?(e.consume(b),C):z(b)}function z(b){return b===61?(e.consume(b),S):Z(b)?(e.consume(b),z):x(b)}function S(b){return b===null||b===60||b===61||b===62||b===96?n(b):b===34||b===39?(e.consume(b),l=b,T):Z(b)?(e.consume(b),S):N(b)}function T(b){return b===l?(e.consume(b),l=null,M):b===null||K(b)?n(b):(e.consume(b),T)}function N(b){return b===null||b===34||b===39||b===47||b===60||b===61||b===62||b===96||ce(b)?z(b):(e.consume(b),N)}function M(b){return b===47||b===62||Z(b)?x(b):n(b)}function w(b){return b===62?(e.consume(b),I):n(b)}function I(b){return b===null||K(b)?W(b):Z(b)?(e.consume(b),I):n(b)}function W(b){return b===45&&i===2?(e.consume(b),$):b===60&&i===1?(e.consume(b),ne):b===62&&i===4?(e.consume(b),B):b===63&&i===3?(e.consume(b),v):b===93&&i===5?(e.consume(b),U):K(b)&&(i===6||i===7)?(e.exit("htmlFlowData"),e.check(Wb,Q,Y)(b)):b===null||K(b)?(e.exit("htmlFlowData"),Y(b)):(e.consume(b),W)}function Y(b){return e.check(Bb,ee,Q)(b)}function ee(b){return e.enter("lineEnding"),e.consume(b),e.exit("lineEnding"),j}function j(b){return b===null||K(b)?Y(b):(e.enter("htmlFlowData"),W(b))}function $(b){return b===45?(e.consume(b),v):W(b)}function ne(b){return b===47?(e.consume(b),o="",O):W(b)}function O(b){if(b===62){const ie=o.toLowerCase();return _p.includes(ie)?(e.consume(b),B):W(b)}return Ue(b)&&o.length<8?(e.consume(b),o+=String.fromCharCode(b),O):W(b)}function U(b){return b===93?(e.consume(b),v):W(b)}function v(b){return b===62?(e.consume(b),B):b===45&&i===2?(e.consume(b),v):W(b)}function B(b){return b===null||K(b)?(e.exit("htmlFlowData"),Q(b)):(e.consume(b),B)}function Q(b){return e.exit("htmlFlow"),t(b)}}function Hb(e,t,n){const r=this;return i;function i(o){return K(o)?(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),a):n(o)}function a(o){return r.parser.lazy[r.now().line]?n(o):t(o)}}function $b(e,t,n){return r;function r(i){return e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),e.attempt(aa,t,n)}}const Qb={name:"htmlText",tokenize:Kb};function Kb(e,t,n){const r=this;let i,a,o;return s;function s(v){return e.enter("htmlText"),e.enter("htmlTextData"),e.consume(v),l}function l(v){return v===33?(e.consume(v),u):v===47?(e.consume(v),z):v===63?(e.consume(v),x):Ue(v)?(e.consume(v),N):n(v)}function u(v){return v===45?(e.consume(v),d):v===91?(e.consume(v),a=0,y):Ue(v)?(e.consume(v),g):n(v)}function d(v){return v===45?(e.consume(v),m):n(v)}function c(v){return v===null?n(v):v===45?(e.consume(v),f):K(v)?(o=c,ne(v)):(e.consume(v),c)}function f(v){return v===45?(e.consume(v),m):c(v)}function m(v){return v===62?$(v):v===45?f(v):c(v)}function y(v){const B="CDATA[";return v===B.charCodeAt(a++)?(e.consume(v),a===B.length?_:y):n(v)}function _(v){return v===null?n(v):v===93?(e.consume(v),k):K(v)?(o=_,ne(v)):(e.consume(v),_)}function k(v){return v===93?(e.consume(v),h):_(v)}function h(v){return v===62?$(v):v===93?(e.consume(v),h):_(v)}function g(v){return v===null||v===62?$(v):K(v)?(o=g,ne(v)):(e.consume(v),g)}function x(v){return v===null?n(v):v===63?(e.consume(v),C):K(v)?(o=x,ne(v)):(e.consume(v),x)}function C(v){return v===62?$(v):x(v)}function z(v){return Ue(v)?(e.consume(v),S):n(v)}function S(v){return v===45||Me(v)?(e.consume(v),S):T(v)}function T(v){return K(v)?(o=T,ne(v)):Z(v)?(e.consume(v),T):$(v)}function N(v){return v===45||Me(v)?(e.consume(v),N):v===47||v===62||ce(v)?M(v):n(v)}function M(v){return v===47?(e.consume(v),$):v===58||v===95||Ue(v)?(e.consume(v),w):K(v)?(o=M,ne(v)):Z(v)?(e.consume(v),M):$(v)}function w(v){return v===45||v===46||v===58||v===95||Me(v)?(e.consume(v),w):I(v)}function I(v){return v===61?(e.consume(v),W):K(v)?(o=I,ne(v)):Z(v)?(e.consume(v),I):M(v)}function W(v){return v===null||v===60||v===61||v===62||v===96?n(v):v===34||v===39?(e.consume(v),i=v,Y):K(v)?(o=W,ne(v)):Z(v)?(e.consume(v),W):(e.consume(v),ee)}function Y(v){return v===i?(e.consume(v),i=void 0,j):v===null?n(v):K(v)?(o=Y,ne(v)):(e.consume(v),Y)}function ee(v){return v===null||v===34||v===39||v===60||v===61||v===96?n(v):v===47||v===62||ce(v)?M(v):(e.consume(v),ee)}function j(v){return v===47||v===62||ce(v)?M(v):n(v)}function $(v){return v===62?(e.consume(v),e.exit("htmlTextData"),e.exit("htmlText"),t):n(v)}function ne(v){return e.exit("htmlTextData"),e.enter("lineEnding"),e.consume(v),e.exit("lineEnding"),O}function O(v){return Z(v)?re(e,U,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(v):U(v)}function U(v){return e.enter("htmlTextData"),o(v)}}const _c={name:"labelEnd",resolveAll:Zb,resolveTo:Jb,tokenize:ew},Yb={tokenize:tw},Xb={tokenize:nw},Gb={tokenize:rw};function Zb(e){let t=-1;const n=[];for(;++t<e.length;){const r=e[t][1];if(n.push(e[t]),r.type==="labelImage"||r.type==="labelLink"||r.type==="labelEnd"){const i=r.type==="labelImage"?4:2;r.type="data",t+=i}}return e.length!==n.length&&lt(e,0,e.length,n),e}function Jb(e,t){let n=e.length,r=0,i,a,o,s;for(;n--;)if(i=e[n][1],a){if(i.type==="link"||i.type==="labelLink"&&i._inactive)break;e[n][0]==="enter"&&i.type==="labelLink"&&(i._inactive=!0)}else if(o){if(e[n][0]==="enter"&&(i.type==="labelImage"||i.type==="labelLink")&&!i._balanced&&(a=n,i.type!=="labelLink")){r=2;break}}else i.type==="labelEnd"&&(o=n);const l={type:e[a][1].type==="labelLink"?"link":"image",start:{...e[a][1].start},end:{...e[e.length-1][1].end}},u={type:"label",start:{...e[a][1].start},end:{...e[o][1].end}},d={type:"labelText",start:{...e[a+r+2][1].end},end:{...e[o-2][1].start}};return s=[["enter",l,t],["enter",u,t]],s=yt(s,e.slice(a+1,a+r+3)),s=yt(s,[["enter",d,t]]),s=yt(s,Xo(t.parser.constructs.insideSpan.null,e.slice(a+r+4,o-3),t)),s=yt(s,[["exit",d,t],e[o-2],e[o-1],["exit",u,t]]),s=yt(s,e.slice(o+1)),s=yt(s,[["exit",l,t]]),lt(e,a,e.length,s),e}function ew(e,t,n){const r=this;let i=r.events.length,a,o;for(;i--;)if((r.events[i][1].type==="labelImage"||r.events[i][1].type==="labelLink")&&!r.events[i][1]._balanced){a=r.events[i][1];break}return s;function s(f){return a?a._inactive?c(f):(o=r.parser.defined.includes(Rt(r.sliceSerialize({start:a.end,end:r.now()}))),e.enter("labelEnd"),e.enter("labelMarker"),e.consume(f),e.exit("labelMarker"),e.exit("labelEnd"),l):n(f)}function l(f){return f===40?e.attempt(Yb,d,o?d:c)(f):f===91?e.attempt(Xb,d,o?u:c)(f):o?d(f):c(f)}function u(f){return e.attempt(Gb,d,c)(f)}function d(f){return t(f)}function c(f){return a._balanced=!0,n(f)}}function tw(e,t,n){return r;function r(c){return e.enter("resource"),e.enter("resourceMarker"),e.consume(c),e.exit("resourceMarker"),i}function i(c){return ce(c)?Si(e,a)(c):a(c)}function a(c){return c===41?d(c):sg(e,o,s,"resourceDestination","resourceDestinationLiteral","resourceDestinationLiteralMarker","resourceDestinationRaw","resourceDestinationString",32)(c)}function o(c){return ce(c)?Si(e,l)(c):d(c)}function s(c){return n(c)}function l(c){return c===34||c===39||c===40?ug(e,u,n,"resourceTitle","resourceTitleMarker","resourceTitleString")(c):d(c)}function u(c){return ce(c)?Si(e,d)(c):d(c)}function d(c){return c===41?(e.enter("resourceMarker"),e.consume(c),e.exit("resourceMarker"),e.exit("resource"),t):n(c)}}function nw(e,t,n){const r=this;return i;function i(s){return lg.call(r,e,a,o,"reference","referenceMarker","referenceString")(s)}function a(s){return r.parser.defined.includes(Rt(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)))?t(s):n(s)}function o(s){return n(s)}}function rw(e,t,n){return r;function r(a){return e.enter("reference"),e.enter("referenceMarker"),e.consume(a),e.exit("referenceMarker"),i}function i(a){return a===93?(e.enter("referenceMarker"),e.consume(a),e.exit("referenceMarker"),e.exit("reference"),t):n(a)}}const iw={name:"labelStartImage",resolveAll:_c.resolveAll,tokenize:aw};function aw(e,t,n){const r=this;return i;function i(s){return e.enter("labelImage"),e.enter("labelImageMarker"),e.consume(s),e.exit("labelImageMarker"),a}function a(s){return s===91?(e.enter("labelMarker"),e.consume(s),e.exit("labelMarker"),e.exit("labelImage"),o):n(s)}function o(s){return s===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(s):t(s)}}const ow={name:"labelStartLink",resolveAll:_c.resolveAll,tokenize:sw};function sw(e,t,n){const r=this;return i;function i(o){return e.enter("labelLink"),e.enter("labelMarker"),e.consume(o),e.exit("labelMarker"),e.exit("labelLink"),a}function a(o){return o===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(o):t(o)}}const Is={name:"lineEnding",tokenize:lw};function lw(e,t){return n;function n(r){return e.enter("lineEnding"),e.consume(r),e.exit("lineEnding"),re(e,t,"linePrefix")}}const Ka={name:"thematicBreak",tokenize:uw};function uw(e,t,n){let r=0,i;return a;function a(u){return e.enter("thematicBreak"),o(u)}function o(u){return i=u,s(u)}function s(u){return u===i?(e.enter("thematicBreakSequence"),l(u)):r>=3&&(u===null||K(u))?(e.exit("thematicBreak"),t(u)):n(u)}function l(u){return u===i?(e.consume(u),r++,l):(e.exit("thematicBreakSequence"),Z(u)?re(e,s,"whitespace")(u):s(u))}}const Ke={continuation:{tokenize:fw},exit:hw,name:"list",tokenize:pw},cw={partial:!0,tokenize:gw},dw={partial:!0,tokenize:mw};function pw(e,t,n){const r=this,i=r.events[r.events.length-1];let a=i&&i[1].type==="linePrefix"?i[2].sliceSerialize(i[1],!0).length:0,o=0;return s;function s(m){const y=r.containerState.type||(m===42||m===43||m===45?"listUnordered":"listOrdered");if(y==="listUnordered"?!r.containerState.marker||m===r.containerState.marker:au(m)){if(r.containerState.type||(r.containerState.type=y,e.enter(y,{_container:!0})),y==="listUnordered")return e.enter("listItemPrefix"),m===42||m===45?e.check(Ka,n,u)(m):u(m);if(!r.interrupt||m===49)return e.enter("listItemPrefix"),e.enter("listItemValue"),l(m)}return n(m)}function l(m){return au(m)&&++o<10?(e.consume(m),l):(!r.interrupt||o<2)&&(r.containerState.marker?m===r.containerState.marker:m===41||m===46)?(e.exit("listItemValue"),u(m)):n(m)}function u(m){return e.enter("listItemMarker"),e.consume(m),e.exit("listItemMarker"),r.containerState.marker=r.containerState.marker||m,e.check(aa,r.interrupt?n:d,e.attempt(cw,f,c))}function d(m){return r.containerState.initialBlankLine=!0,a++,f(m)}function c(m){return Z(m)?(e.enter("listItemPrefixWhitespace"),e.consume(m),e.exit("listItemPrefixWhitespace"),f):n(m)}function f(m){return r.containerState.size=a+r.sliceSerialize(e.exit("listItemPrefix"),!0).length,t(m)}}function fw(e,t,n){const r=this;return r.containerState._closeFlow=void 0,e.check(aa,i,a);function i(s){return r.containerState.furtherBlankLines=r.containerState.furtherBlankLines||r.containerState.initialBlankLine,re(e,t,"listItemIndent",r.containerState.size+1)(s)}function a(s){return r.containerState.furtherBlankLines||!Z(s)?(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,o(s)):(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,e.attempt(dw,t,o)(s))}function o(s){return r.containerState._closeFlow=!0,r.interrupt=void 0,re(e,e.attempt(Ke,t,n),"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(s)}}function mw(e,t,n){const r=this;return re(e,i,"listItemIndent",r.containerState.size+1);function i(a){const o=r.events[r.events.length-1];return o&&o[1].type==="listItemIndent"&&o[2].sliceSerialize(o[1],!0).length===r.containerState.size?t(a):n(a)}}function hw(e){e.exit(this.containerState.type)}function gw(e,t,n){const r=this;return re(e,i,"listItemPrefixWhitespace",r.parser.constructs.disable.null.includes("codeIndented")?void 0:5);function i(a){const o=r.events[r.events.length-1];return!Z(a)&&o&&o[1].type==="listItemPrefixWhitespace"?t(a):n(a)}}const vp={name:"setextUnderline",resolveTo:yw,tokenize:xw};function yw(e,t){let n=e.length,r,i,a;for(;n--;)if(e[n][0]==="enter"){if(e[n][1].type==="content"){r=n;break}e[n][1].type==="paragraph"&&(i=n)}else e[n][1].type==="content"&&e.splice(n,1),!a&&e[n][1].type==="definition"&&(a=n);const o={type:"setextHeading",start:{...e[r][1].start},end:{...e[e.length-1][1].end}};return e[i][1].type="setextHeadingText",a?(e.splice(i,0,["enter",o,t]),e.splice(a+1,0,["exit",e[r][1],t]),e[r][1].end={...e[a][1].end}):e[r][1]=o,e.push(["exit",o,t]),e}function xw(e,t,n){const r=this;let i;return a;function a(u){let d=r.events.length,c;for(;d--;)if(r.events[d][1].type!=="lineEnding"&&r.events[d][1].type!=="linePrefix"&&r.events[d][1].type!=="content"){c=r.events[d][1].type==="paragraph";break}return!r.parser.lazy[r.now().line]&&(r.interrupt||c)?(e.enter("setextHeadingLine"),i=u,o(u)):n(u)}function o(u){return e.enter("setextHeadingLineSequence"),s(u)}function s(u){return u===i?(e.consume(u),s):(e.exit("setextHeadingLineSequence"),Z(u)?re(e,l,"lineSuffix")(u):l(u))}function l(u){return u===null||K(u)?(e.exit("setextHeadingLine"),t(u)):n(u)}}const _w={tokenize:vw};function vw(e){const t=this,n=e.attempt(aa,r,e.attempt(this.parser.constructs.flowInitial,i,re(e,e.attempt(this.parser.constructs.flow,i,e.attempt(Cb,i)),"linePrefix")));return n;function r(a){if(a===null){e.consume(a);return}return e.enter("lineEndingBlank"),e.consume(a),e.exit("lineEndingBlank"),t.currentConstruct=void 0,n}function i(a){if(a===null){e.consume(a);return}return e.enter("lineEnding"),e.consume(a),e.exit("lineEnding"),t.currentConstruct=void 0,n}}const bw={resolveAll:dg()},ww=cg("string"),kw=cg("text");function cg(e){return{resolveAll:dg(e==="text"?Sw:void 0),tokenize:t};function t(n){const r=this,i=this.parser.constructs[e],a=n.attempt(i,o,s);return o;function o(d){return u(d)?a(d):s(d)}function s(d){if(d===null){n.consume(d);return}return n.enter("data"),n.consume(d),l}function l(d){return u(d)?(n.exit("data"),a(d)):(n.consume(d),l)}function u(d){if(d===null)return!0;const c=i[d];let f=-1;if(c)for(;++f<c.length;){const m=c[f];if(!m.previous||m.previous.call(r,r.previous))return!0}return!1}}}function dg(e){return t;function t(n,r){let i=-1,a;for(;++i<=n.length;)a===void 0?n[i]&&n[i][1].type==="data"&&(a=i,i++):(!n[i]||n[i][1].type!=="data")&&(i!==a+2&&(n[a][1].end=n[i-1][1].end,n.splice(a+2,i-a-2),i=a+2),a=void 0);return e?e(n,r):n}}function Sw(e,t){let n=0;for(;++n<=e.length;)if((n===e.length||e[n][1].type==="lineEnding")&&e[n-1][1].type==="data"){const r=e[n-1][1],i=t.sliceStream(r);let a=i.length,o=-1,s=0,l;for(;a--;){const u=i[a];if(typeof u=="string"){for(o=u.length;u.charCodeAt(o-1)===32;)s++,o--;if(o)break;o=-1}else if(u===-2)l=!0,s++;else if(u!==-1){a++;break}}if(t._contentTypeTextTrailing&&n===e.length&&(s=0),s){const u={type:n===e.length||l||s<2?"lineSuffix":"hardBreakTrailing",start:{_bufferIndex:a?o:r.start._bufferIndex+o,_index:r.start._index+a,line:r.end.line,column:r.end.column-s,offset:r.end.offset-s},end:{...r.end}};r.end={...u.start},r.start.offset===r.end.offset?Object.assign(r,u):(e.splice(n,0,["enter",u,t],["exit",u,t]),n+=2)}n++}return e}const Cw={42:Ke,43:Ke,45:Ke,48:Ke,49:Ke,50:Ke,51:Ke,52:Ke,53:Ke,54:Ke,55:Ke,56:Ke,57:Ke,62:rg},Ew={91:Nb},zw={[-2]:Rs,[-1]:Rs,32:Rs},Tw={35:Ob,42:Ka,45:[vp,Ka],60:qb,61:vp,95:Ka,96:xp,126:xp},Pw={38:ag,92:ig},Nw={[-5]:Is,[-4]:Is,[-3]:Is,33:iw,38:ag,42:ou,60:[ob,Qb],91:ow,92:[Rb,ig],93:_c,95:ou,96:_b},jw={null:[ou,bw]},Lw={null:[42,95]},Aw={null:[]},Rw=Object.freeze(Object.defineProperty({__proto__:null,attentionMarkers:Lw,contentInitial:Ew,disable:Aw,document:Cw,flow:Tw,flowInitial:zw,insideSpan:jw,string:Pw,text:Nw},Symbol.toStringTag,{value:"Module"}));function Iw(e,t,n){let r={_bufferIndex:-1,_index:0,line:n&&n.line||1,column:n&&n.column||1,offset:n&&n.offset||0};const i={},a=[];let o=[],s=[];const l={attempt:T(z),check:T(S),consume:g,enter:x,exit:C,interrupt:T(S,{interrupt:!0})},u={code:null,containerState:{},defineSkip:_,events:[],now:y,parser:e,previous:null,sliceSerialize:f,sliceStream:m,write:c};let d=t.tokenize.call(u,l);return t.resolveAll&&a.push(t),u;function c(I){return o=yt(o,I),k(),o[o.length-1]!==null?[]:(N(t,0),u.events=Xo(a,u.events,u),u.events)}function f(I,W){return Dw(m(I),W)}function m(I){return Ow(o,I)}function y(){const{_bufferIndex:I,_index:W,line:Y,column:ee,offset:j}=r;return{_bufferIndex:I,_index:W,line:Y,column:ee,offset:j}}function _(I){i[I.line]=I.column,w()}function k(){let I;for(;r._index<o.length;){const W=o[r._index];if(typeof W=="string")for(I=r._index,r._bufferIndex<0&&(r._bufferIndex=0);r._index===I&&r._bufferIndex<W.length;)h(W.charCodeAt(r._bufferIndex));else h(W)}}function h(I){d=d(I)}function g(I){K(I)?(r.line++,r.column=1,r.offset+=I===-3?2:1,w()):I!==-1&&(r.column++,r.offset++),r._bufferIndex<0?r._index++:(r._bufferIndex++,r._bufferIndex===o[r._index].length&&(r._bufferIndex=-1,r._index++)),u.previous=I}function x(I,W){const Y=W||{};return Y.type=I,Y.start=y(),u.events.push(["enter",Y,u]),s.push(Y),Y}function C(I){const W=s.pop();return W.end=y(),u.events.push(["exit",W,u]),W}function z(I,W){N(I,W.from)}function S(I,W){W.restore()}function T(I,W){return Y;function Y(ee,j,$){let ne,O,U,v;return Array.isArray(ee)?Q(ee):"tokenize"in ee?Q([ee]):B(ee);function B(le){return pt;function pt(q){const L=q!==null&&le[q],D=q!==null&&le.null,H=[...Array.isArray(L)?L:L?[L]:[],...Array.isArray(D)?D:D?[D]:[]];return Q(H)(q)}}function Q(le){return ne=le,O=0,le.length===0?$:b(le[O])}function b(le){return pt;function pt(q){return v=M(),U=le,le.partial||(u.currentConstruct=le),le.name&&u.parser.constructs.disable.null.includes(le.name)?we():le.tokenize.call(W?Object.assign(Object.create(u),W):u,l,ie,we)(q)}}function ie(le){return I(U,v),j}function we(le){return v.restore(),++O<ne.length?b(ne[O]):$}}}function N(I,W){I.resolveAll&&!a.includes(I)&&a.push(I),I.resolve&&lt(u.events,W,u.events.length-W,I.resolve(u.events.slice(W),u)),I.resolveTo&&(u.events=I.resolveTo(u.events,u))}function M(){const I=y(),W=u.previous,Y=u.currentConstruct,ee=u.events.length,j=Array.from(s);return{from:ee,restore:$};function $(){r=I,u.previous=W,u.currentConstruct=Y,u.events.length=ee,s=j,w()}}function w(){r.line in i&&r.column<2&&(r.column=i[r.line],r.offset+=i[r.line]-1)}}function Ow(e,t){const n=t.start._index,r=t.start._bufferIndex,i=t.end._index,a=t.end._bufferIndex;let o;if(n===i)o=[e[n].slice(r,a)];else{if(o=e.slice(n,i),r>-1){const s=o[0];typeof s=="string"?o[0]=s.slice(r):o.shift()}a>0&&o.push(e[i].slice(0,a))}return o}function Dw(e,t){let n=-1;const r=[];let i;for(;++n<e.length;){const a=e[n];let o;if(typeof a=="string")o=a;else switch(a){case-5:{o="\r";break}case-4:{o=`
`;break}case-3:{o=`\r
`;break}case-2:{o=t?" ":"	";break}case-1:{if(!t&&i)continue;o=" ";break}default:o=String.fromCharCode(a)}i=a===-2,r.push(o)}return r.join("")}function Fw(e){const r={constructs:tg([Rw,...(e||{}).extensions||[]]),content:i(Jv),defined:[],document:i(tb),flow:i(_w),lazy:{},string:i(ww),text:i(kw)};return r;function i(a){return o;function o(s){return Iw(r,a,s)}}}function Mw(e){for(;!og(e););return e}const bp=/[\0\t\n\r]/g;function qw(){let e=1,t="",n=!0,r;return i;function i(a,o,s){const l=[];let u,d,c,f,m;for(a=t+(typeof a=="string"?a.toString():new TextDecoder(o||void 0).decode(a)),c=0,t="",n&&(a.charCodeAt(0)===65279&&c++,n=void 0);c<a.length;){if(bp.lastIndex=c,u=bp.exec(a),f=u&&u.index!==void 0?u.index:a.length,m=a.charCodeAt(f),!u){t=a.slice(c);break}if(m===10&&c===f&&r)l.push(-3),r=void 0;else switch(r&&(l.push(-5),r=void 0),c<f&&(l.push(a.slice(c,f)),e+=f-c),m){case 0:{l.push(65533),e++;break}case 9:{for(d=Math.ceil(e/4)*4,l.push(-2);e++<d;)l.push(-1);break}case 10:{l.push(-4),e=1;break}default:r=!0,e=1}c=f+1}return s&&(r&&l.push(-5),t&&l.push(t),l.push(null)),l}}const Ww=/\\([!-/:-@[-`{-~])|&(#(?:\d{1,7}|x[\da-f]{1,6})|[\da-z]{1,31});/gi;function Bw(e){return e.replace(Ww,Vw)}function Vw(e,t,n){if(t)return t;if(n.charCodeAt(0)===35){const i=n.charCodeAt(1),a=i===120||i===88;return ng(n.slice(a?2:1),a?16:10)}return xc(n)||e}const pg={}.hasOwnProperty;function Uw(e,t,n){return typeof t!="string"&&(n=t,t=void 0),Hw(n)(Mw(Fw(n).document().write(qw()(e,t,!0))))}function Hw(e){const t={transforms:[],canContainEols:["emphasis","fragment","heading","paragraph","strong"],enter:{autolink:a(Nc),autolinkProtocol:M,autolinkEmail:M,atxHeading:a(Ot),blockQuote:a(D),characterEscape:M,characterReference:M,codeFenced:a(H),codeFencedFenceInfo:o,codeFencedFenceMeta:o,codeIndented:a(H,o),codeText:a(Ee,o),codeTextData:M,data:M,codeFlowValue:M,definition:a(_e),definitionDestinationString:o,definitionLabelString:o,definitionTitleString:o,emphasis:a(Be),hardBreakEscape:a(An),hardBreakTrailing:a(An),htmlFlow:a(Rn,o),htmlFlowData:M,htmlText:a(Rn,o),htmlTextData:M,image:a(ly),label:o,link:a(Nc),listItem:a(uy),listItemValue:f,listOrdered:a(jc,c),listUnordered:a(jc),paragraph:a(cy),reference:b,referenceString:o,resourceDestinationString:o,resourceTitleString:o,setextHeading:a(Ot),strong:a(dy),thematicBreak:a(fy)},exit:{atxHeading:l(),atxHeadingSequence:z,autolink:l(),autolinkEmail:L,autolinkProtocol:q,blockQuote:l(),characterEscapeValue:w,characterReferenceMarkerHexadecimal:we,characterReferenceMarkerNumeric:we,characterReferenceValue:le,characterReference:pt,codeFenced:l(k),codeFencedFence:_,codeFencedFenceInfo:m,codeFencedFenceMeta:y,codeFlowValue:w,codeIndented:l(h),codeText:l(j),codeTextData:w,data:w,definition:l(),definitionDestinationString:C,definitionLabelString:g,definitionTitleString:x,emphasis:l(),hardBreakEscape:l(W),hardBreakTrailing:l(W),htmlFlow:l(Y),htmlFlowData:w,htmlText:l(ee),htmlTextData:w,image:l(ne),label:U,labelText:O,lineEnding:I,link:l($),listItem:l(),listOrdered:l(),listUnordered:l(),paragraph:l(),referenceString:ie,resourceDestinationString:v,resourceTitleString:B,resource:Q,setextHeading:l(N),setextHeadingLineSequence:T,setextHeadingText:S,strong:l(),thematicBreak:l()}};fg(t,(e||{}).mdastExtensions||[]);const n={};return r;function r(P){let F={type:"root",children:[]};const X={stack:[F],tokenStack:[],config:t,enter:s,exit:u,buffer:o,resume:d,data:n},te=[];let ue=-1;for(;++ue<P.length;)if(P[ue][1].type==="listOrdered"||P[ue][1].type==="listUnordered")if(P[ue][0]==="enter")te.push(ue);else{const wt=te.pop();ue=i(P,wt,ue)}for(ue=-1;++ue<P.length;){const wt=t[P[ue][0]];pg.call(wt,P[ue][1].type)&&wt[P[ue][1].type].call(Object.assign({sliceSerialize:P[ue][2].sliceSerialize},X),P[ue][1])}if(X.tokenStack.length>0){const wt=X.tokenStack[X.tokenStack.length-1];(wt[1]||wp).call(X,void 0,wt[0])}for(F.position={start:sn(P.length>0?P[0][1].start:{line:1,column:1,offset:0}),end:sn(P.length>0?P[P.length-2][1].end:{line:1,column:1,offset:0})},ue=-1;++ue<t.transforms.length;)F=t.transforms[ue](F)||F;return F}function i(P,F,X){let te=F-1,ue=-1,wt=!1,In,Ut,Kr,Yr;for(;++te<=X;){const tt=P[te];switch(tt[1].type){case"listUnordered":case"listOrdered":case"blockQuote":{tt[0]==="enter"?ue++:ue--,Yr=void 0;break}case"lineEndingBlank":{tt[0]==="enter"&&(In&&!Yr&&!ue&&!Kr&&(Kr=te),Yr=void 0);break}case"linePrefix":case"listItemValue":case"listItemMarker":case"listItemPrefix":case"listItemPrefixWhitespace":break;default:Yr=void 0}if(!ue&&tt[0]==="enter"&&tt[1].type==="listItemPrefix"||ue===-1&&tt[0]==="exit"&&(tt[1].type==="listUnordered"||tt[1].type==="listOrdered")){if(In){let rr=te;for(Ut=void 0;rr--;){const Ht=P[rr];if(Ht[1].type==="lineEnding"||Ht[1].type==="lineEndingBlank"){if(Ht[0]==="exit")continue;Ut&&(P[Ut][1].type="lineEndingBlank",wt=!0),Ht[1].type="lineEnding",Ut=rr}else if(!(Ht[1].type==="linePrefix"||Ht[1].type==="blockQuotePrefix"||Ht[1].type==="blockQuotePrefixWhitespace"||Ht[1].type==="blockQuoteMarker"||Ht[1].type==="listItemIndent"))break}Kr&&(!Ut||Kr<Ut)&&(In._spread=!0),In.end=Object.assign({},Ut?P[Ut][1].start:tt[1].end),P.splice(Ut||te,0,["exit",In,tt[2]]),te++,X++}if(tt[1].type==="listItemPrefix"){const rr={type:"listItem",_spread:!1,start:Object.assign({},tt[1].start),end:void 0};In=rr,P.splice(te,0,["enter",rr,tt[2]]),te++,X++,Kr=void 0,Yr=!0}}}return P[F][1]._spread=wt,X}function a(P,F){return X;function X(te){s.call(this,P(te),te),F&&F.call(this,te)}}function o(){this.stack.push({type:"fragment",children:[]})}function s(P,F,X){this.stack[this.stack.length-1].children.push(P),this.stack.push(P),this.tokenStack.push([F,X||void 0]),P.position={start:sn(F.start),end:void 0}}function l(P){return F;function F(X){P&&P.call(this,X),u.call(this,X)}}function u(P,F){const X=this.stack.pop(),te=this.tokenStack.pop();if(te)te[0].type!==P.type&&(F?F.call(this,P,te[0]):(te[1]||wp).call(this,P,te[0]));else throw new Error("Cannot close `"+P.type+"` ("+ki({start:P.start,end:P.end})+"): it’s not open");X.position.end=sn(P.end)}function d(){return yc(this.stack.pop())}function c(){this.data.expectingFirstListItemValue=!0}function f(P){if(this.data.expectingFirstListItemValue){const F=this.stack[this.stack.length-2];F.start=Number.parseInt(this.sliceSerialize(P),10),this.data.expectingFirstListItemValue=void 0}}function m(){const P=this.resume(),F=this.stack[this.stack.length-1];F.lang=P}function y(){const P=this.resume(),F=this.stack[this.stack.length-1];F.meta=P}function _(){this.data.flowCodeInside||(this.buffer(),this.data.flowCodeInside=!0)}function k(){const P=this.resume(),F=this.stack[this.stack.length-1];F.value=P.replace(/^(\r?\n|\r)|(\r?\n|\r)$/g,""),this.data.flowCodeInside=void 0}function h(){const P=this.resume(),F=this.stack[this.stack.length-1];F.value=P.replace(/(\r?\n|\r)$/g,"")}function g(P){const F=this.resume(),X=this.stack[this.stack.length-1];X.label=F,X.identifier=Rt(this.sliceSerialize(P)).toLowerCase()}function x(){const P=this.resume(),F=this.stack[this.stack.length-1];F.title=P}function C(){const P=this.resume(),F=this.stack[this.stack.length-1];F.url=P}function z(P){const F=this.stack[this.stack.length-1];if(!F.depth){const X=this.sliceSerialize(P).length;F.depth=X}}function S(){this.data.setextHeadingSlurpLineEnding=!0}function T(P){const F=this.stack[this.stack.length-1];F.depth=this.sliceSerialize(P).codePointAt(0)===61?1:2}function N(){this.data.setextHeadingSlurpLineEnding=void 0}function M(P){const X=this.stack[this.stack.length-1].children;let te=X[X.length-1];(!te||te.type!=="text")&&(te=py(),te.position={start:sn(P.start),end:void 0},X.push(te)),this.stack.push(te)}function w(P){const F=this.stack.pop();F.value+=this.sliceSerialize(P),F.position.end=sn(P.end)}function I(P){const F=this.stack[this.stack.length-1];if(this.data.atHardBreak){const X=F.children[F.children.length-1];X.position.end=sn(P.end),this.data.atHardBreak=void 0;return}!this.data.setextHeadingSlurpLineEnding&&t.canContainEols.includes(F.type)&&(M.call(this,P),w.call(this,P))}function W(){this.data.atHardBreak=!0}function Y(){const P=this.resume(),F=this.stack[this.stack.length-1];F.value=P}function ee(){const P=this.resume(),F=this.stack[this.stack.length-1];F.value=P}function j(){const P=this.resume(),F=this.stack[this.stack.length-1];F.value=P}function $(){const P=this.stack[this.stack.length-1];if(this.data.inReference){const F=this.data.referenceType||"shortcut";P.type+="Reference",P.referenceType=F,delete P.url,delete P.title}else delete P.identifier,delete P.label;this.data.referenceType=void 0}function ne(){const P=this.stack[this.stack.length-1];if(this.data.inReference){const F=this.data.referenceType||"shortcut";P.type+="Reference",P.referenceType=F,delete P.url,delete P.title}else delete P.identifier,delete P.label;this.data.referenceType=void 0}function O(P){const F=this.sliceSerialize(P),X=this.stack[this.stack.length-2];X.label=Bw(F),X.identifier=Rt(F).toLowerCase()}function U(){const P=this.stack[this.stack.length-1],F=this.resume(),X=this.stack[this.stack.length-1];if(this.data.inReference=!0,X.type==="link"){const te=P.children;X.children=te}else X.alt=F}function v(){const P=this.resume(),F=this.stack[this.stack.length-1];F.url=P}function B(){const P=this.resume(),F=this.stack[this.stack.length-1];F.title=P}function Q(){this.data.inReference=void 0}function b(){this.data.referenceType="collapsed"}function ie(P){const F=this.resume(),X=this.stack[this.stack.length-1];X.label=F,X.identifier=Rt(this.sliceSerialize(P)).toLowerCase(),this.data.referenceType="full"}function we(P){this.data.characterReferenceType=P.type}function le(P){const F=this.sliceSerialize(P),X=this.data.characterReferenceType;let te;X?(te=ng(F,X==="characterReferenceMarkerNumeric"?10:16),this.data.characterReferenceType=void 0):te=xc(F);const ue=this.stack[this.stack.length-1];ue.value+=te}function pt(P){const F=this.stack.pop();F.position.end=sn(P.end)}function q(P){w.call(this,P);const F=this.stack[this.stack.length-1];F.url=this.sliceSerialize(P)}function L(P){w.call(this,P);const F=this.stack[this.stack.length-1];F.url="mailto:"+this.sliceSerialize(P)}function D(){return{type:"blockquote",children:[]}}function H(){return{type:"code",lang:null,meta:null,value:""}}function Ee(){return{type:"inlineCode",value:""}}function _e(){return{type:"definition",identifier:"",label:null,title:null,url:""}}function Be(){return{type:"emphasis",children:[]}}function Ot(){return{type:"heading",depth:0,children:[]}}function An(){return{type:"break"}}function Rn(){return{type:"html",value:""}}function ly(){return{type:"image",title:null,url:"",alt:null}}function Nc(){return{type:"link",title:null,url:"",children:[]}}function jc(P){return{type:"list",ordered:P.type==="listOrdered",start:null,spread:P._spread,children:[]}}function uy(P){return{type:"listItem",spread:P._spread,checked:null,children:[]}}function cy(){return{type:"paragraph",children:[]}}function dy(){return{type:"strong",children:[]}}function py(){return{type:"text",value:""}}function fy(){return{type:"thematicBreak"}}}function sn(e){return{line:e.line,column:e.column,offset:e.offset}}function fg(e,t){let n=-1;for(;++n<t.length;){const r=t[n];Array.isArray(r)?fg(e,r):$w(e,r)}}function $w(e,t){let n;for(n in t)if(pg.call(t,n))switch(n){case"canContainEols":{const r=t[n];r&&e[n].push(...r);break}case"transforms":{const r=t[n];r&&e[n].push(...r);break}case"enter":case"exit":{const r=t[n];r&&Object.assign(e[n],r);break}}}function wp(e,t){throw e?new Error("Cannot close `"+e.type+"` ("+ki({start:e.start,end:e.end})+"): a different token (`"+t.type+"`, "+ki({start:t.start,end:t.end})+") is open"):new Error("Cannot close document, a token (`"+t.type+"`, "+ki({start:t.start,end:t.end})+") is still open")}function Qw(e){const t=this;t.parser=n;function n(r){return Uw(r,{...t.data("settings"),...e,extensions:t.data("micromarkExtensions")||[],mdastExtensions:t.data("fromMarkdownExtensions")||[]})}}function Kw(e,t){const n={type:"element",tagName:"blockquote",properties:{},children:e.wrap(e.all(t),!0)};return e.patch(t,n),e.applyData(t,n)}function Yw(e,t){const n={type:"element",tagName:"br",properties:{},children:[]};return e.patch(t,n),[e.applyData(t,n),{type:"text",value:`
`}]}function Xw(e,t){const n=t.value?t.value+`
`:"",r={},i=t.lang?t.lang.split(/\s+/):[];i.length>0&&(r.className=["language-"+i[0]]);let a={type:"element",tagName:"code",properties:r,children:[{type:"text",value:n}]};return t.meta&&(a.data={meta:t.meta}),e.patch(t,a),a=e.applyData(t,a),a={type:"element",tagName:"pre",properties:{},children:[a]},e.patch(t,a),a}function Gw(e,t){const n={type:"element",tagName:"del",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function Zw(e,t){const n={type:"element",tagName:"em",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function Jw(e,t){const n=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",r=String(t.identifier).toUpperCase(),i=Qr(r.toLowerCase()),a=e.footnoteOrder.indexOf(r);let o,s=e.footnoteCounts.get(r);s===void 0?(s=0,e.footnoteOrder.push(r),o=e.footnoteOrder.length):o=a+1,s+=1,e.footnoteCounts.set(r,s);const l={type:"element",tagName:"a",properties:{href:"#"+n+"fn-"+i,id:n+"fnref-"+i+(s>1?"-"+s:""),dataFootnoteRef:!0,ariaDescribedBy:["footnote-label"]},children:[{type:"text",value:String(o)}]};e.patch(t,l);const u={type:"element",tagName:"sup",properties:{},children:[l]};return e.patch(t,u),e.applyData(t,u)}function ek(e,t){const n={type:"element",tagName:"h"+t.depth,properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function tk(e,t){if(e.options.allowDangerousHtml){const n={type:"raw",value:t.value};return e.patch(t,n),e.applyData(t,n)}}function mg(e,t){const n=t.referenceType;let r="]";if(n==="collapsed"?r+="[]":n==="full"&&(r+="["+(t.label||t.identifier)+"]"),t.type==="imageReference")return[{type:"text",value:"!["+t.alt+r}];const i=e.all(t),a=i[0];a&&a.type==="text"?a.value="["+a.value:i.unshift({type:"text",value:"["});const o=i[i.length-1];return o&&o.type==="text"?o.value+=r:i.push({type:"text",value:r}),i}function nk(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return mg(e,t);const i={src:Qr(r.url||""),alt:t.alt};r.title!==null&&r.title!==void 0&&(i.title=r.title);const a={type:"element",tagName:"img",properties:i,children:[]};return e.patch(t,a),e.applyData(t,a)}function rk(e,t){const n={src:Qr(t.url)};t.alt!==null&&t.alt!==void 0&&(n.alt=t.alt),t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"img",properties:n,children:[]};return e.patch(t,r),e.applyData(t,r)}function ik(e,t){const n={type:"text",value:t.value.replace(/\r?\n|\r/g," ")};e.patch(t,n);const r={type:"element",tagName:"code",properties:{},children:[n]};return e.patch(t,r),e.applyData(t,r)}function ak(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return mg(e,t);const i={href:Qr(r.url||"")};r.title!==null&&r.title!==void 0&&(i.title=r.title);const a={type:"element",tagName:"a",properties:i,children:e.all(t)};return e.patch(t,a),e.applyData(t,a)}function ok(e,t){const n={href:Qr(t.url)};t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"a",properties:n,children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function sk(e,t,n){const r=e.all(t),i=n?lk(n):hg(t),a={},o=[];if(typeof t.checked=="boolean"){const d=r[0];let c;d&&d.type==="element"&&d.tagName==="p"?c=d:(c={type:"element",tagName:"p",properties:{},children:[]},r.unshift(c)),c.children.length>0&&c.children.unshift({type:"text",value:" "}),c.children.unshift({type:"element",tagName:"input",properties:{type:"checkbox",checked:t.checked,disabled:!0},children:[]}),a.className=["task-list-item"]}let s=-1;for(;++s<r.length;){const d=r[s];(i||s!==0||d.type!=="element"||d.tagName!=="p")&&o.push({type:"text",value:`
`}),d.type==="element"&&d.tagName==="p"&&!i?o.push(...d.children):o.push(d)}const l=r[r.length-1];l&&(i||l.type!=="element"||l.tagName!=="p")&&o.push({type:"text",value:`
`});const u={type:"element",tagName:"li",properties:a,children:o};return e.patch(t,u),e.applyData(t,u)}function lk(e){let t=!1;if(e.type==="list"){t=e.spread||!1;const n=e.children;let r=-1;for(;!t&&++r<n.length;)t=hg(n[r])}return t}function hg(e){const t=e.spread;return t??e.children.length>1}function uk(e,t){const n={},r=e.all(t);let i=-1;for(typeof t.start=="number"&&t.start!==1&&(n.start=t.start);++i<r.length;){const o=r[i];if(o.type==="element"&&o.tagName==="li"&&o.properties&&Array.isArray(o.properties.className)&&o.properties.className.includes("task-list-item")){n.className=["contains-task-list"];break}}const a={type:"element",tagName:t.ordered?"ol":"ul",properties:n,children:e.wrap(r,!0)};return e.patch(t,a),e.applyData(t,a)}function ck(e,t){const n={type:"element",tagName:"p",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function dk(e,t){const n={type:"root",children:e.wrap(e.all(t))};return e.patch(t,n),e.applyData(t,n)}function pk(e,t){const n={type:"element",tagName:"strong",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function fk(e,t){const n=e.all(t),r=n.shift(),i=[];if(r){const o={type:"element",tagName:"thead",properties:{},children:e.wrap([r],!0)};e.patch(t.children[0],o),i.push(o)}if(n.length>0){const o={type:"element",tagName:"tbody",properties:{},children:e.wrap(n,!0)},s=fc(t.children[1]),l=Kh(t.children[t.children.length-1]);s&&l&&(o.position={start:s,end:l}),i.push(o)}const a={type:"element",tagName:"table",properties:{},children:e.wrap(i,!0)};return e.patch(t,a),e.applyData(t,a)}function mk(e,t,n){const r=n?n.children:void 0,a=(r?r.indexOf(t):1)===0?"th":"td",o=n&&n.type==="table"?n.align:void 0,s=o?o.length:t.children.length;let l=-1;const u=[];for(;++l<s;){const c=t.children[l],f={},m=o?o[l]:void 0;m&&(f.align=m);let y={type:"element",tagName:a,properties:f,children:[]};c&&(y.children=e.all(c),e.patch(c,y),y=e.applyData(c,y)),u.push(y)}const d={type:"element",tagName:"tr",properties:{},children:e.wrap(u,!0)};return e.patch(t,d),e.applyData(t,d)}function hk(e,t){const n={type:"element",tagName:"td",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}const kp=9,Sp=32;function gk(e){const t=String(e),n=/\r?\n|\r/g;let r=n.exec(t),i=0;const a=[];for(;r;)a.push(Cp(t.slice(i,r.index),i>0,!0),r[0]),i=r.index+r[0].length,r=n.exec(t);return a.push(Cp(t.slice(i),i>0,!1)),a.join("")}function Cp(e,t,n){let r=0,i=e.length;if(t){let a=e.codePointAt(r);for(;a===kp||a===Sp;)r++,a=e.codePointAt(r)}if(n){let a=e.codePointAt(i-1);for(;a===kp||a===Sp;)i--,a=e.codePointAt(i-1)}return i>r?e.slice(r,i):""}function yk(e,t){const n={type:"text",value:gk(String(t.value))};return e.patch(t,n),e.applyData(t,n)}function xk(e,t){const n={type:"element",tagName:"hr",properties:{},children:[]};return e.patch(t,n),e.applyData(t,n)}const _k={blockquote:Kw,break:Yw,code:Xw,delete:Gw,emphasis:Zw,footnoteReference:Jw,heading:ek,html:tk,imageReference:nk,image:rk,inlineCode:ik,linkReference:ak,link:ok,listItem:sk,list:uk,paragraph:ck,root:dk,strong:pk,table:fk,tableCell:hk,tableRow:mk,text:yk,thematicBreak:xk,toml:Ea,yaml:Ea,definition:Ea,footnoteDefinition:Ea};function Ea(){}const gg=-1,Go=0,Ci=1,Co=2,vc=3,bc=4,wc=5,kc=6,yg=7,xg=8,Ep=typeof self=="object"?self:globalThis,vk=(e,t)=>{const n=(i,a)=>(e.set(a,i),i),r=i=>{if(e.has(i))return e.get(i);const[a,o]=t[i];switch(a){case Go:case gg:return n(o,i);case Ci:{const s=n([],i);for(const l of o)s.push(r(l));return s}case Co:{const s=n({},i);for(const[l,u]of o)s[r(l)]=r(u);return s}case vc:return n(new Date(o),i);case bc:{const{source:s,flags:l}=o;return n(new RegExp(s,l),i)}case wc:{const s=n(new Map,i);for(const[l,u]of o)s.set(r(l),r(u));return s}case kc:{const s=n(new Set,i);for(const l of o)s.add(r(l));return s}case yg:{const{name:s,message:l}=o;return n(new Ep[s](l),i)}case xg:return n(BigInt(o),i);case"BigInt":return n(Object(BigInt(o)),i);case"ArrayBuffer":return n(new Uint8Array(o).buffer,o);case"DataView":{const{buffer:s}=new Uint8Array(o);return n(new DataView(s),o)}}return n(new Ep[a](o),i)};return r},zp=e=>vk(new Map,e)(0),or="",{toString:bk}={},{keys:wk}=Object,si=e=>{const t=typeof e;if(t!=="object"||!e)return[Go,t];const n=bk.call(e).slice(8,-1);switch(n){case"Array":return[Ci,or];case"Object":return[Co,or];case"Date":return[vc,or];case"RegExp":return[bc,or];case"Map":return[wc,or];case"Set":return[kc,or];case"DataView":return[Ci,n]}return n.includes("Array")?[Ci,n]:n.includes("Error")?[yg,n]:[Co,n]},za=([e,t])=>e===Go&&(t==="function"||t==="symbol"),kk=(e,t,n,r)=>{const i=(o,s)=>{const l=r.push(o)-1;return n.set(s,l),l},a=o=>{if(n.has(o))return n.get(o);let[s,l]=si(o);switch(s){case Go:{let d=o;switch(l){case"bigint":s=xg,d=o.toString();break;case"function":case"symbol":if(e)throw new TypeError("unable to serialize "+l);d=null;break;case"undefined":return i([gg],o)}return i([s,d],o)}case Ci:{if(l){let f=o;return l==="DataView"?f=new Uint8Array(o.buffer):l==="ArrayBuffer"&&(f=new Uint8Array(o)),i([l,[...f]],o)}const d=[],c=i([s,d],o);for(const f of o)d.push(a(f));return c}case Co:{if(l)switch(l){case"BigInt":return i([l,o.toString()],o);case"Boolean":case"Number":case"String":return i([l,o.valueOf()],o)}if(t&&"toJSON"in o)return a(o.toJSON());const d=[],c=i([s,d],o);for(const f of wk(o))(e||!za(si(o[f])))&&d.push([a(f),a(o[f])]);return c}case vc:return i([s,o.toISOString()],o);case bc:{const{source:d,flags:c}=o;return i([s,{source:d,flags:c}],o)}case wc:{const d=[],c=i([s,d],o);for(const[f,m]of o)(e||!(za(si(f))||za(si(m))))&&d.push([a(f),a(m)]);return c}case kc:{const d=[],c=i([s,d],o);for(const f of o)(e||!za(si(f)))&&d.push(a(f));return c}}const{message:u}=o;return i([s,{name:l,message:u}],o)};return a},Tp=(e,{json:t,lossy:n}={})=>{const r=[];return kk(!(t||n),!!t,new Map,r)(e),r},Eo=typeof structuredClone=="function"?(e,t)=>t&&("json"in t||"lossy"in t)?zp(Tp(e,t)):structuredClone(e):(e,t)=>zp(Tp(e,t));function Sk(e,t){const n=[{type:"text",value:"↩"}];return t>1&&n.push({type:"element",tagName:"sup",properties:{},children:[{type:"text",value:String(t)}]}),n}function Ck(e,t){return"Back to reference "+(e+1)+(t>1?"-"+t:"")}function Ek(e){const t=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",n=e.options.footnoteBackContent||Sk,r=e.options.footnoteBackLabel||Ck,i=e.options.footnoteLabel||"Footnotes",a=e.options.footnoteLabelTagName||"h2",o=e.options.footnoteLabelProperties||{className:["sr-only"]},s=[];let l=-1;for(;++l<e.footnoteOrder.length;){const u=e.footnoteById.get(e.footnoteOrder[l]);if(!u)continue;const d=e.all(u),c=String(u.identifier).toUpperCase(),f=Qr(c.toLowerCase());let m=0;const y=[],_=e.footnoteCounts.get(c);for(;_!==void 0&&++m<=_;){y.length>0&&y.push({type:"text",value:" "});let g=typeof n=="string"?n:n(l,m);typeof g=="string"&&(g={type:"text",value:g}),y.push({type:"element",tagName:"a",properties:{href:"#"+t+"fnref-"+f+(m>1?"-"+m:""),dataFootnoteBackref:"",ariaLabel:typeof r=="string"?r:r(l,m),className:["data-footnote-backref"]},children:Array.isArray(g)?g:[g]})}const k=d[d.length-1];if(k&&k.type==="element"&&k.tagName==="p"){const g=k.children[k.children.length-1];g&&g.type==="text"?g.value+=" ":k.children.push({type:"text",value:" "}),k.children.push(...y)}else d.push(...y);const h={type:"element",tagName:"li",properties:{id:t+"fn-"+f},children:e.wrap(d,!0)};e.patch(u,h),s.push(h)}if(s.length!==0)return{type:"element",tagName:"section",properties:{dataFootnotes:!0,className:["footnotes"]},children:[{type:"element",tagName:a,properties:{...Eo(o),id:"footnote-label"},children:[{type:"text",value:i}]},{type:"text",value:`
`},{type:"element",tagName:"ol",properties:{},children:e.wrap(s,!0)},{type:"text",value:`
`}]}}const Zo=function(e){if(e==null)return Nk;if(typeof e=="function")return Jo(e);if(typeof e=="object")return Array.isArray(e)?zk(e):Tk(e);if(typeof e=="string")return Pk(e);throw new Error("Expected function, string, or object as test")};function zk(e){const t=[];let n=-1;for(;++n<e.length;)t[n]=Zo(e[n]);return Jo(r);function r(...i){let a=-1;for(;++a<t.length;)if(t[a].apply(this,i))return!0;return!1}}function Tk(e){const t=e;return Jo(n);function n(r){const i=r;let a;for(a in e)if(i[a]!==t[a])return!1;return!0}}function Pk(e){return Jo(t);function t(n){return n&&n.type===e}}function Jo(e){return t;function t(n,r,i){return!!(jk(n)&&e.call(this,n,typeof r=="number"?r:void 0,i||void 0))}}function Nk(){return!0}function jk(e){return e!==null&&typeof e=="object"&&"type"in e}const _g=[],Lk=!0,su=!1,Ak="skip";function vg(e,t,n,r){let i;typeof t=="function"&&typeof n!="function"?(r=n,n=t):i=t;const a=Zo(i),o=r?-1:1;s(e,void 0,[])();function s(l,u,d){const c=l&&typeof l=="object"?l:{};if(typeof c.type=="string"){const m=typeof c.tagName=="string"?c.tagName:typeof c.name=="string"?c.name:void 0;Object.defineProperty(f,"name",{value:"node ("+(l.type+(m?"<"+m+">":""))+")"})}return f;function f(){let m=_g,y,_,k;if((!t||a(l,u,d[d.length-1]||void 0))&&(m=Rk(n(l,d)),m[0]===su))return m;if("children"in l&&l.children){const h=l;if(h.children&&m[0]!==Ak)for(_=(r?h.children.length:-1)+o,k=d.concat(h);_>-1&&_<h.children.length;){const g=h.children[_];if(y=s(g,_,k)(),y[0]===su)return y;_=typeof y[1]=="number"?y[1]:_+o}}return m}}}function Rk(e){return Array.isArray(e)?e:typeof e=="number"?[Lk,e]:e==null?_g:[e]}function Sc(e,t,n,r){let i,a,o;typeof t=="function"&&typeof n!="function"?(a=void 0,o=t,i=n):(a=t,o=n,i=r),vg(e,a,s,i);function s(l,u){const d=u[u.length-1],c=d?d.children.indexOf(l):void 0;return o(l,c,d)}}const lu={}.hasOwnProperty,Ik={};function Ok(e,t){const n=t||Ik,r=new Map,i=new Map,a=new Map,o={..._k,...n.handlers},s={all:u,applyData:Fk,definitionById:r,footnoteById:i,footnoteCounts:a,footnoteOrder:[],handlers:o,one:l,options:n,patch:Dk,wrap:qk};return Sc(e,function(d){if(d.type==="definition"||d.type==="footnoteDefinition"){const c=d.type==="definition"?r:i,f=String(d.identifier).toUpperCase();c.has(f)||c.set(f,d)}}),s;function l(d,c){const f=d.type,m=s.handlers[f];if(lu.call(s.handlers,f)&&m)return m(s,d,c);if(s.options.passThrough&&s.options.passThrough.includes(f)){if("children"in d){const{children:_,...k}=d,h=Eo(k);return h.children=s.all(d),h}return Eo(d)}return(s.options.unknownHandler||Mk)(s,d,c)}function u(d){const c=[];if("children"in d){const f=d.children;let m=-1;for(;++m<f.length;){const y=s.one(f[m],d);if(y){if(m&&f[m-1].type==="break"&&(!Array.isArray(y)&&y.type==="text"&&(y.value=Pp(y.value)),!Array.isArray(y)&&y.type==="element")){const _=y.children[0];_&&_.type==="text"&&(_.value=Pp(_.value))}Array.isArray(y)?c.push(...y):c.push(y)}}}return c}}function Dk(e,t){e.position&&(t.position=Cv(e))}function Fk(e,t){let n=t;if(e&&e.data){const r=e.data.hName,i=e.data.hChildren,a=e.data.hProperties;if(typeof r=="string")if(n.type==="element")n.tagName=r;else{const o="children"in n?n.children:[n];n={type:"element",tagName:r,properties:{},children:o}}n.type==="element"&&a&&Object.assign(n.properties,Eo(a)),"children"in n&&n.children&&i!==null&&i!==void 0&&(n.children=i)}return n}function Mk(e,t){const n=t.data||{},r="value"in t&&!(lu.call(n,"hProperties")||lu.call(n,"hChildren"))?{type:"text",value:t.value}:{type:"element",tagName:"div",properties:{},children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function qk(e,t){const n=[];let r=-1;for(t&&n.push({type:"text",value:`
`});++r<e.length;)r&&n.push({type:"text",value:`
`}),n.push(e[r]);return t&&e.length>0&&n.push({type:"text",value:`
`}),n}function Pp(e){let t=0,n=e.charCodeAt(t);for(;n===9||n===32;)t++,n=e.charCodeAt(t);return e.slice(t)}function Np(e,t){const n=Ok(e,t),r=n.one(e,void 0),i=Ek(n),a=Array.isArray(r)?{type:"root",children:r}:r||{type:"root",children:[]};return i&&a.children.push({type:"text",value:`
`},i),a}function Wk(e,t){return e&&"run"in e?async function(n,r){const i=Np(n,{file:r,...t});await e.run(i,r)}:function(n,r){return Np(n,{file:r,...e||t})}}function jp(e){if(e)throw e}var Ya=Object.prototype.hasOwnProperty,bg=Object.prototype.toString,Lp=Object.defineProperty,Ap=Object.getOwnPropertyDescriptor,Rp=function(t){return typeof Array.isArray=="function"?Array.isArray(t):bg.call(t)==="[object Array]"},Ip=function(t){if(!t||bg.call(t)!=="[object Object]")return!1;var n=Ya.call(t,"constructor"),r=t.constructor&&t.constructor.prototype&&Ya.call(t.constructor.prototype,"isPrototypeOf");if(t.constructor&&!n&&!r)return!1;var i;for(i in t);return typeof i>"u"||Ya.call(t,i)},Op=function(t,n){Lp&&n.name==="__proto__"?Lp(t,n.name,{enumerable:!0,configurable:!0,value:n.newValue,writable:!0}):t[n.name]=n.newValue},Dp=function(t,n){if(n==="__proto__")if(Ya.call(t,n)){if(Ap)return Ap(t,n).value}else return;return t[n]},Bk=function e(){var t,n,r,i,a,o,s=arguments[0],l=1,u=arguments.length,d=!1;for(typeof s=="boolean"&&(d=s,s=arguments[1]||{},l=2),(s==null||typeof s!="object"&&typeof s!="function")&&(s={});l<u;++l)if(t=arguments[l],t!=null)for(n in t)r=Dp(s,n),i=Dp(t,n),s!==i&&(d&&i&&(Ip(i)||(a=Rp(i)))?(a?(a=!1,o=r&&Rp(r)?r:[]):o=r&&Ip(r)?r:{},Op(s,{name:n,newValue:e(d,o,i)})):typeof i<"u"&&Op(s,{name:n,newValue:i}));return s};const Os=Jn(Bk);function uu(e){if(typeof e!="object"||e===null)return!1;const t=Object.getPrototypeOf(e);return(t===null||t===Object.prototype||Object.getPrototypeOf(t)===null)&&!(Symbol.toStringTag in e)&&!(Symbol.iterator in e)}function Vk(){const e=[],t={run:n,use:r};return t;function n(...i){let a=-1;const o=i.pop();if(typeof o!="function")throw new TypeError("Expected function as last argument, not "+o);s(null,...i);function s(l,...u){const d=e[++a];let c=-1;if(l){o(l);return}for(;++c<i.length;)(u[c]===null||u[c]===void 0)&&(u[c]=i[c]);i=u,d?Uk(d,s)(...u):o(null,...u)}}function r(i){if(typeof i!="function")throw new TypeError("Expected `middelware` to be a function, not "+i);return e.push(i),t}}function Uk(e,t){let n;return r;function r(...o){const s=e.length>o.length;let l;s&&o.push(i);try{l=e.apply(this,o)}catch(u){const d=u;if(s&&n)throw d;return i(d)}s||(l&&l.then&&typeof l.then=="function"?l.then(a,i):l instanceof Error?i(l):a(l))}function i(o,...s){n||(n=!0,t(o,...s))}function a(o){i(null,o)}}const Mt={basename:Hk,dirname:$k,extname:Qk,join:Kk,sep:"/"};function Hk(e,t){if(t!==void 0&&typeof t!="string")throw new TypeError('"ext" argument must be a string');oa(e);let n=0,r=-1,i=e.length,a;if(t===void 0||t.length===0||t.length>e.length){for(;i--;)if(e.codePointAt(i)===47){if(a){n=i+1;break}}else r<0&&(a=!0,r=i+1);return r<0?"":e.slice(n,r)}if(t===e)return"";let o=-1,s=t.length-1;for(;i--;)if(e.codePointAt(i)===47){if(a){n=i+1;break}}else o<0&&(a=!0,o=i+1),s>-1&&(e.codePointAt(i)===t.codePointAt(s--)?s<0&&(r=i):(s=-1,r=o));return n===r?r=o:r<0&&(r=e.length),e.slice(n,r)}function $k(e){if(oa(e),e.length===0)return".";let t=-1,n=e.length,r;for(;--n;)if(e.codePointAt(n)===47){if(r){t=n;break}}else r||(r=!0);return t<0?e.codePointAt(0)===47?"/":".":t===1&&e.codePointAt(0)===47?"//":e.slice(0,t)}function Qk(e){oa(e);let t=e.length,n=-1,r=0,i=-1,a=0,o;for(;t--;){const s=e.codePointAt(t);if(s===47){if(o){r=t+1;break}continue}n<0&&(o=!0,n=t+1),s===46?i<0?i=t:a!==1&&(a=1):i>-1&&(a=-1)}return i<0||n<0||a===0||a===1&&i===n-1&&i===r+1?"":e.slice(i,n)}function Kk(...e){let t=-1,n;for(;++t<e.length;)oa(e[t]),e[t]&&(n=n===void 0?e[t]:n+"/"+e[t]);return n===void 0?".":Yk(n)}function Yk(e){oa(e);const t=e.codePointAt(0)===47;let n=Xk(e,!t);return n.length===0&&!t&&(n="."),n.length>0&&e.codePointAt(e.length-1)===47&&(n+="/"),t?"/"+n:n}function Xk(e,t){let n="",r=0,i=-1,a=0,o=-1,s,l;for(;++o<=e.length;){if(o<e.length)s=e.codePointAt(o);else{if(s===47)break;s=47}if(s===47){if(!(i===o-1||a===1))if(i!==o-1&&a===2){if(n.length<2||r!==2||n.codePointAt(n.length-1)!==46||n.codePointAt(n.length-2)!==46){if(n.length>2){if(l=n.lastIndexOf("/"),l!==n.length-1){l<0?(n="",r=0):(n=n.slice(0,l),r=n.length-1-n.lastIndexOf("/")),i=o,a=0;continue}}else if(n.length>0){n="",r=0,i=o,a=0;continue}}t&&(n=n.length>0?n+"/..":"..",r=2)}else n.length>0?n+="/"+e.slice(i+1,o):n=e.slice(i+1,o),r=o-i-1;i=o,a=0}else s===46&&a>-1?a++:a=-1}return n}function oa(e){if(typeof e!="string")throw new TypeError("Path must be a string. Received "+JSON.stringify(e))}const Gk={cwd:Zk};function Zk(){return"/"}function cu(e){return!!(e!==null&&typeof e=="object"&&"href"in e&&e.href&&"protocol"in e&&e.protocol&&e.auth===void 0)}function Jk(e){if(typeof e=="string")e=new URL(e);else if(!cu(e)){const t=new TypeError('The "path" argument must be of type string or an instance of URL. Received `'+e+"`");throw t.code="ERR_INVALID_ARG_TYPE",t}if(e.protocol!=="file:"){const t=new TypeError("The URL must be of scheme file");throw t.code="ERR_INVALID_URL_SCHEME",t}return e2(e)}function e2(e){if(e.hostname!==""){const r=new TypeError('File URL host must be "localhost" or empty on darwin');throw r.code="ERR_INVALID_FILE_URL_HOST",r}const t=e.pathname;let n=-1;for(;++n<t.length;)if(t.codePointAt(n)===37&&t.codePointAt(n+1)===50){const r=t.codePointAt(n+2);if(r===70||r===102){const i=new TypeError("File URL path must not include encoded / characters");throw i.code="ERR_INVALID_FILE_URL_PATH",i}}return decodeURIComponent(t)}const Ds=["history","path","basename","stem","extname","dirname"];class wg{constructor(t){let n;t?cu(t)?n={path:t}:typeof t=="string"||t2(t)?n={value:t}:n=t:n={},this.cwd="cwd"in n?"":Gk.cwd(),this.data={},this.history=[],this.messages=[],this.value,this.map,this.result,this.stored;let r=-1;for(;++r<Ds.length;){const a=Ds[r];a in n&&n[a]!==void 0&&n[a]!==null&&(this[a]=a==="history"?[...n[a]]:n[a])}let i;for(i in n)Ds.includes(i)||(this[i]=n[i])}get basename(){return typeof this.path=="string"?Mt.basename(this.path):void 0}set basename(t){Ms(t,"basename"),Fs(t,"basename"),this.path=Mt.join(this.dirname||"",t)}get dirname(){return typeof this.path=="string"?Mt.dirname(this.path):void 0}set dirname(t){Fp(this.basename,"dirname"),this.path=Mt.join(t||"",this.basename)}get extname(){return typeof this.path=="string"?Mt.extname(this.path):void 0}set extname(t){if(Fs(t,"extname"),Fp(this.dirname,"extname"),t){if(t.codePointAt(0)!==46)throw new Error("`extname` must start with `.`");if(t.includes(".",1))throw new Error("`extname` cannot contain multiple dots")}this.path=Mt.join(this.dirname,this.stem+(t||""))}get path(){return this.history[this.history.length-1]}set path(t){cu(t)&&(t=Jk(t)),Ms(t,"path"),this.path!==t&&this.history.push(t)}get stem(){return typeof this.path=="string"?Mt.basename(this.path,this.extname):void 0}set stem(t){Ms(t,"stem"),Fs(t,"stem"),this.path=Mt.join(this.dirname||"",t+(this.extname||""))}fail(t,n,r){const i=this.message(t,n,r);throw i.fatal=!0,i}info(t,n,r){const i=this.message(t,n,r);return i.fatal=void 0,i}message(t,n,r){const i=new We(t,n,r);return this.path&&(i.name=this.path+":"+i.name,i.file=this.path),i.fatal=!1,this.messages.push(i),i}toString(t){return this.value===void 0?"":typeof this.value=="string"?this.value:new TextDecoder(t||void 0).decode(this.value)}}function Fs(e,t){if(e&&e.includes(Mt.sep))throw new Error("`"+t+"` cannot be a path: did not expect `"+Mt.sep+"`")}function Ms(e,t){if(!e)throw new Error("`"+t+"` cannot be empty")}function Fp(e,t){if(!e)throw new Error("Setting `"+t+"` requires `path` to be set too")}function t2(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const n2=function(e){const r=this.constructor.prototype,i=r[e],a=function(){return i.apply(a,arguments)};return Object.setPrototypeOf(a,r),a},r2={}.hasOwnProperty;class Cc extends n2{constructor(){super("copy"),this.Compiler=void 0,this.Parser=void 0,this.attachers=[],this.compiler=void 0,this.freezeIndex=-1,this.frozen=void 0,this.namespace={},this.parser=void 0,this.transformers=Vk()}copy(){const t=new Cc;let n=-1;for(;++n<this.attachers.length;){const r=this.attachers[n];t.use(...r)}return t.data(Os(!0,{},this.namespace)),t}data(t,n){return typeof t=="string"?arguments.length===2?(Bs("data",this.frozen),this.namespace[t]=n,this):r2.call(this.namespace,t)&&this.namespace[t]||void 0:t?(Bs("data",this.frozen),this.namespace=t,this):this.namespace}freeze(){if(this.frozen)return this;const t=this;for(;++this.freezeIndex<this.attachers.length;){const[n,...r]=this.attachers[this.freezeIndex];if(r[0]===!1)continue;r[0]===!0&&(r[0]=void 0);const i=n.call(t,...r);typeof i=="function"&&this.transformers.use(i)}return this.frozen=!0,this.freezeIndex=Number.POSITIVE_INFINITY,this}parse(t){this.freeze();const n=Ta(t),r=this.parser||this.Parser;return qs("parse",r),r(String(n),n)}process(t,n){const r=this;return this.freeze(),qs("process",this.parser||this.Parser),Ws("process",this.compiler||this.Compiler),n?i(void 0,n):new Promise(i);function i(a,o){const s=Ta(t),l=r.parse(s);r.run(l,s,function(d,c,f){if(d||!c||!f)return u(d);const m=c,y=r.stringify(m,f);o2(y)?f.value=y:f.result=y,u(d,f)});function u(d,c){d||!c?o(d):a?a(c):n(void 0,c)}}}processSync(t){let n=!1,r;return this.freeze(),qs("processSync",this.parser||this.Parser),Ws("processSync",this.compiler||this.Compiler),this.process(t,i),qp("processSync","process",n),r;function i(a,o){n=!0,jp(a),r=o}}run(t,n,r){Mp(t),this.freeze();const i=this.transformers;return!r&&typeof n=="function"&&(r=n,n=void 0),r?a(void 0,r):new Promise(a);function a(o,s){const l=Ta(n);i.run(t,l,u);function u(d,c,f){const m=c||t;d?s(d):o?o(m):r(void 0,m,f)}}}runSync(t,n){let r=!1,i;return this.run(t,n,a),qp("runSync","run",r),i;function a(o,s){jp(o),i=s,r=!0}}stringify(t,n){this.freeze();const r=Ta(n),i=this.compiler||this.Compiler;return Ws("stringify",i),Mp(t),i(t,r)}use(t,...n){const r=this.attachers,i=this.namespace;if(Bs("use",this.frozen),t!=null)if(typeof t=="function")l(t,n);else if(typeof t=="object")Array.isArray(t)?s(t):o(t);else throw new TypeError("Expected usable value, not `"+t+"`");return this;function a(u){if(typeof u=="function")l(u,[]);else if(typeof u=="object")if(Array.isArray(u)){const[d,...c]=u;l(d,c)}else o(u);else throw new TypeError("Expected usable value, not `"+u+"`")}function o(u){if(!("plugins"in u)&&!("settings"in u))throw new Error("Expected usable value but received an empty preset, which is probably a mistake: presets typically come with `plugins` and sometimes with `settings`, but this has neither");s(u.plugins),u.settings&&(i.settings=Os(!0,i.settings,u.settings))}function s(u){let d=-1;if(u!=null)if(Array.isArray(u))for(;++d<u.length;){const c=u[d];a(c)}else throw new TypeError("Expected a list of plugins, not `"+u+"`")}function l(u,d){let c=-1,f=-1;for(;++c<r.length;)if(r[c][0]===u){f=c;break}if(f===-1)r.push([u,...d]);else if(d.length>0){let[m,...y]=d;const _=r[f][1];uu(_)&&uu(m)&&(m=Os(!0,_,m)),r[f]=[u,m,...y]}}}}const i2=new Cc().freeze();function qs(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `parser`")}function Ws(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `compiler`")}function Bs(e,t){if(t)throw new Error("Cannot call `"+e+"` on a frozen processor.\nCreate a new processor first, by calling it: use `processor()` instead of `processor`.")}function Mp(e){if(!uu(e)||typeof e.type!="string")throw new TypeError("Expected node, got `"+e+"`")}function qp(e,t,n){if(!n)throw new Error("`"+e+"` finished async. Use `"+t+"` instead")}function Ta(e){return a2(e)?e:new wg(e)}function a2(e){return!!(e&&typeof e=="object"&&"message"in e&&"messages"in e)}function o2(e){return typeof e=="string"||s2(e)}function s2(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const l2="https://github.com/remarkjs/react-markdown/blob/main/changelog.md",Wp=[],Bp={allowDangerousHtml:!0},u2=/^(https?|ircs?|mailto|xmpp)$/i,c2=[{from:"astPlugins",id:"remove-buggy-html-in-markdown-parser"},{from:"allowDangerousHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"allowNode",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowElement"},{from:"allowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowedElements"},{from:"disallowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"disallowedElements"},{from:"escapeHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"includeElementIndex",id:"#remove-includeelementindex"},{from:"includeNodeIndex",id:"change-includenodeindex-to-includeelementindex"},{from:"linkTarget",id:"remove-linktarget"},{from:"plugins",id:"change-plugins-to-remarkplugins",to:"remarkPlugins"},{from:"rawSourcePos",id:"#remove-rawsourcepos"},{from:"renderers",id:"change-renderers-to-components",to:"components"},{from:"source",id:"change-source-to-children",to:"children"},{from:"sourcePos",id:"#remove-sourcepos"},{from:"transformImageUri",id:"#add-urltransform",to:"urlTransform"},{from:"transformLinkUri",id:"#add-urltransform",to:"urlTransform"}];function kg(e){const t=d2(e),n=p2(e);return f2(t.runSync(t.parse(n),n),e)}function d2(e){const t=e.rehypePlugins||Wp,n=e.remarkPlugins||Wp,r=e.remarkRehypeOptions?{...e.remarkRehypeOptions,...Bp}:Bp;return i2().use(Qw).use(n).use(Wk,r).use(t)}function p2(e){const t=e.children||"",n=new wg;return typeof t=="string"&&(n.value=t),n}function f2(e,t){const n=t.allowedElements,r=t.allowElement,i=t.components,a=t.disallowedElements,o=t.skipHtml,s=t.unwrapDisallowed,l=t.urlTransform||m2;for(const d of c2)Object.hasOwn(t,d.from)&&(""+d.from+(d.to?"use `"+d.to+"` instead":"remove it")+l2+d.id,void 0);return t.className&&(e={type:"element",tagName:"div",properties:{className:t.className},children:e.type==="root"?e.children:[e]}),Sc(e,u),Nv(e,{Fragment:p.Fragment,components:i,ignoreInvalidStyle:!0,jsx:p.jsx,jsxs:p.jsxs,passKeys:!0,passNode:!0});function u(d,c,f){if(d.type==="raw"&&f&&typeof c=="number")return o?f.children.splice(c,1):f.children[c]={type:"text",value:d.value},c;if(d.type==="element"){let m;for(m in As)if(Object.hasOwn(As,m)&&Object.hasOwn(d.properties,m)){const y=d.properties[m],_=As[m];(_===null||_.includes(d.tagName))&&(d.properties[m]=l(String(y||""),m,d))}}if(d.type==="element"){let m=n?!n.includes(d.tagName):a?a.includes(d.tagName):!1;if(!m&&r&&typeof c=="number"&&(m=!r(d,c,f)),m&&f&&typeof c=="number")return s&&d.children?f.children.splice(c,1,...d.children):f.children.splice(c,1),c}}}function m2(e){const t=e.indexOf(":"),n=e.indexOf("?"),r=e.indexOf("#"),i=e.indexOf("/");return t===-1||i!==-1&&t>i||n!==-1&&t>n||r!==-1&&t>r||u2.test(e.slice(0,t))?e:""}function Vp(e,t){const n=String(e);if(typeof t!="string")throw new TypeError("Expected character");let r=0,i=n.indexOf(t);for(;i!==-1;)r++,i=n.indexOf(t,i+t.length);return r}function h2(e){if(typeof e!="string")throw new TypeError("Expected a string");return e.replace(/[|\\{}()[\]^$+*?.]/g,"\\$&").replace(/-/g,"\\x2d")}function g2(e,t,n){const i=Zo((n||{}).ignore||[]),a=y2(t);let o=-1;for(;++o<a.length;)vg(e,"text",s);function s(u,d){let c=-1,f;for(;++c<d.length;){const m=d[c],y=f?f.children:void 0;if(i(m,y?y.indexOf(m):void 0,f))return;f=m}if(f)return l(u,d)}function l(u,d){const c=d[d.length-1],f=a[o][0],m=a[o][1];let y=0;const k=c.children.indexOf(u);let h=!1,g=[];f.lastIndex=0;let x=f.exec(u.value);for(;x;){const C=x.index,z={index:x.index,input:x.input,stack:[...d,u]};let S=m(...x,z);if(typeof S=="string"&&(S=S.length>0?{type:"text",value:S}:void 0),S===!1?f.lastIndex=C+1:(y!==C&&g.push({type:"text",value:u.value.slice(y,C)}),Array.isArray(S)?g.push(...S):S&&g.push(S),y=C+x[0].length,h=!0),!f.global)break;x=f.exec(u.value)}return h?(y<u.value.length&&g.push({type:"text",value:u.value.slice(y)}),c.children.splice(k,1,...g)):g=[u],k+g.length}}function y2(e){const t=[];if(!Array.isArray(e))throw new TypeError("Expected find and replace tuple or list of tuples");const n=!e[0]||Array.isArray(e[0])?e:[e];let r=-1;for(;++r<n.length;){const i=n[r];t.push([x2(i[0]),_2(i[1])])}return t}function x2(e){return typeof e=="string"?new RegExp(h2(e),"g"):e}function _2(e){return typeof e=="function"?e:function(){return e}}const Vs="phrasing",Us=["autolink","link","image","label"];function v2(){return{transforms:[z2],enter:{literalAutolink:w2,literalAutolinkEmail:Hs,literalAutolinkHttp:Hs,literalAutolinkWww:Hs},exit:{literalAutolink:E2,literalAutolinkEmail:C2,literalAutolinkHttp:k2,literalAutolinkWww:S2}}}function b2(){return{unsafe:[{character:"@",before:"[+\\-.\\w]",after:"[\\-.\\w]",inConstruct:Vs,notInConstruct:Us},{character:".",before:"[Ww]",after:"[\\-.\\w]",inConstruct:Vs,notInConstruct:Us},{character:":",before:"[ps]",after:"\\/",inConstruct:Vs,notInConstruct:Us}]}}function w2(e){this.enter({type:"link",title:null,url:"",children:[]},e)}function Hs(e){this.config.enter.autolinkProtocol.call(this,e)}function k2(e){this.config.exit.autolinkProtocol.call(this,e)}function S2(e){this.config.exit.data.call(this,e);const t=this.stack[this.stack.length-1];t.type,t.url="http://"+this.sliceSerialize(e)}function C2(e){this.config.exit.autolinkEmail.call(this,e)}function E2(e){this.exit(e)}function z2(e){g2(e,[[/(https?:\/\/|www(?=\.))([-.\w]+)([^ \t\r\n]*)/gi,T2],[new RegExp("(?<=^|\\s|\\p{P}|\\p{S})([-.\\w+]+)@([-\\w]+(?:\\.[-\\w]+)+)","gu"),P2]],{ignore:["link","linkReference"]})}function T2(e,t,n,r,i){let a="";if(!Sg(i)||(/^w/i.test(t)&&(n=t+n,t="",a="http://"),!N2(n)))return!1;const o=j2(n+r);if(!o[0])return!1;const s={type:"link",title:null,url:a+t+o[0],children:[{type:"text",value:t+o[0]}]};return o[1]?[s,{type:"text",value:o[1]}]:s}function P2(e,t,n,r){return!Sg(r,!0)||/[-\d_]$/.test(n)?!1:{type:"link",title:null,url:"mailto:"+t+"@"+n,children:[{type:"text",value:t+"@"+n}]}}function N2(e){const t=e.split(".");return!(t.length<2||t[t.length-1]&&(/_/.test(t[t.length-1])||!/[a-zA-Z\d]/.test(t[t.length-1]))||t[t.length-2]&&(/_/.test(t[t.length-2])||!/[a-zA-Z\d]/.test(t[t.length-2])))}function j2(e){const t=/[!"&'),.:;<>?\]}]+$/.exec(e);if(!t)return[e,void 0];e=e.slice(0,t.index);let n=t[0],r=n.indexOf(")");const i=Vp(e,"(");let a=Vp(e,")");for(;r!==-1&&i>a;)e+=n.slice(0,r+1),n=n.slice(r+1),r=n.indexOf(")"),a++;return[e,n]}function Sg(e,t){const n=e.input.charCodeAt(e.index-1);return(e.index===0||Zn(n)||Yo(n))&&(!t||n!==47)}Cg.peek=q2;function L2(){this.buffer()}function A2(e){this.enter({type:"footnoteReference",identifier:"",label:""},e)}function R2(){this.buffer()}function I2(e){this.enter({type:"footnoteDefinition",identifier:"",label:"",children:[]},e)}function O2(e){const t=this.resume(),n=this.stack[this.stack.length-1];n.type,n.identifier=Rt(this.sliceSerialize(e)).toLowerCase(),n.label=t}function D2(e){this.exit(e)}function F2(e){const t=this.resume(),n=this.stack[this.stack.length-1];n.type,n.identifier=Rt(this.sliceSerialize(e)).toLowerCase(),n.label=t}function M2(e){this.exit(e)}function q2(){return"["}function Cg(e,t,n,r){const i=n.createTracker(r);let a=i.move("[^");const o=n.enter("footnoteReference"),s=n.enter("reference");return a+=i.move(n.safe(n.associationId(e),{after:"]",before:a})),s(),o(),a+=i.move("]"),a}function W2(){return{enter:{gfmFootnoteCallString:L2,gfmFootnoteCall:A2,gfmFootnoteDefinitionLabelString:R2,gfmFootnoteDefinition:I2},exit:{gfmFootnoteCallString:O2,gfmFootnoteCall:D2,gfmFootnoteDefinitionLabelString:F2,gfmFootnoteDefinition:M2}}}function B2(e){let t=!1;return e&&e.firstLineBlank&&(t=!0),{handlers:{footnoteDefinition:n,footnoteReference:Cg},unsafe:[{character:"[",inConstruct:["label","phrasing","reference"]}]};function n(r,i,a,o){const s=a.createTracker(o);let l=s.move("[^");const u=a.enter("footnoteDefinition"),d=a.enter("label");return l+=s.move(a.safe(a.associationId(r),{before:l,after:"]"})),d(),l+=s.move("]:"),r.children&&r.children.length>0&&(s.shift(4),l+=s.move((t?`
`:" ")+a.indentLines(a.containerFlow(r,s.current()),t?Eg:V2))),u(),l}}function V2(e,t,n){return t===0?e:Eg(e,t,n)}function Eg(e,t,n){return(n?"":"    ")+e}const U2=["autolink","destinationLiteral","destinationRaw","reference","titleQuote","titleApostrophe"];zg.peek=Y2;function H2(){return{canContainEols:["delete"],enter:{strikethrough:Q2},exit:{strikethrough:K2}}}function $2(){return{unsafe:[{character:"~",inConstruct:"phrasing",notInConstruct:U2}],handlers:{delete:zg}}}function Q2(e){this.enter({type:"delete",children:[]},e)}function K2(e){this.exit(e)}function zg(e,t,n,r){const i=n.createTracker(r),a=n.enter("strikethrough");let o=i.move("~~");return o+=n.containerPhrasing(e,{...i.current(),before:o,after:"~"}),o+=i.move("~~"),a(),o}function Y2(){return"~"}function X2(e){return e.length}function G2(e,t){const n=t||{},r=(n.align||[]).concat(),i=n.stringLength||X2,a=[],o=[],s=[],l=[];let u=0,d=-1;for(;++d<e.length;){const _=[],k=[];let h=-1;for(e[d].length>u&&(u=e[d].length);++h<e[d].length;){const g=Z2(e[d][h]);if(n.alignDelimiters!==!1){const x=i(g);k[h]=x,(l[h]===void 0||x>l[h])&&(l[h]=x)}_.push(g)}o[d]=_,s[d]=k}let c=-1;if(typeof r=="object"&&"length"in r)for(;++c<u;)a[c]=Up(r[c]);else{const _=Up(r);for(;++c<u;)a[c]=_}c=-1;const f=[],m=[];for(;++c<u;){const _=a[c];let k="",h="";_===99?(k=":",h=":"):_===108?k=":":_===114&&(h=":");let g=n.alignDelimiters===!1?1:Math.max(1,l[c]-k.length-h.length);const x=k+"-".repeat(g)+h;n.alignDelimiters!==!1&&(g=k.length+g+h.length,g>l[c]&&(l[c]=g),m[c]=g),f[c]=x}o.splice(1,0,f),s.splice(1,0,m),d=-1;const y=[];for(;++d<o.length;){const _=o[d],k=s[d];c=-1;const h=[];for(;++c<u;){const g=_[c]||"";let x="",C="";if(n.alignDelimiters!==!1){const z=l[c]-(k[c]||0),S=a[c];S===114?x=" ".repeat(z):S===99?z%2?(x=" ".repeat(z/2+.5),C=" ".repeat(z/2-.5)):(x=" ".repeat(z/2),C=x):C=" ".repeat(z)}n.delimiterStart!==!1&&!c&&h.push("|"),n.padding!==!1&&!(n.alignDelimiters===!1&&g==="")&&(n.delimiterStart!==!1||c)&&h.push(" "),n.alignDelimiters!==!1&&h.push(x),h.push(g),n.alignDelimiters!==!1&&h.push(C),n.padding!==!1&&h.push(" "),(n.delimiterEnd!==!1||c!==u-1)&&h.push("|")}y.push(n.delimiterEnd===!1?h.join("").replace(/ +$/,""):h.join(""))}return y.join(`
`)}function Z2(e){return e==null?"":String(e)}function Up(e){const t=typeof e=="string"?e.codePointAt(0):0;return t===67||t===99?99:t===76||t===108?108:t===82||t===114?114:0}function J2(e,t,n,r){const i=n.enter("blockquote"),a=n.createTracker(r);a.move("> "),a.shift(2);const o=n.indentLines(n.containerFlow(e,a.current()),eS);return i(),o}function eS(e,t,n){return">"+(n?"":" ")+e}function tS(e,t){return Hp(e,t.inConstruct,!0)&&!Hp(e,t.notInConstruct,!1)}function Hp(e,t,n){if(typeof t=="string"&&(t=[t]),!t||t.length===0)return n;let r=-1;for(;++r<t.length;)if(e.includes(t[r]))return!0;return!1}function $p(e,t,n,r){let i=-1;for(;++i<n.unsafe.length;)if(n.unsafe[i].character===`
`&&tS(n.stack,n.unsafe[i]))return/[ \t]/.test(r.before)?"":" ";return`\\
`}function nS(e,t){const n=String(e);let r=n.indexOf(t),i=r,a=0,o=0;if(typeof t!="string")throw new TypeError("Expected substring");for(;r!==-1;)r===i?++a>o&&(o=a):a=1,i=r+t.length,r=n.indexOf(t,i);return o}function rS(e,t){return!!(t.options.fences===!1&&e.value&&!e.lang&&/[^ \r\n]/.test(e.value)&&!/^[\t ]*(?:[\r\n]|$)|(?:^|[\r\n])[\t ]*$/.test(e.value))}function iS(e){const t=e.options.fence||"`";if(t!=="`"&&t!=="~")throw new Error("Cannot serialize code with `"+t+"` for `options.fence`, expected `` ` `` or `~`");return t}function aS(e,t,n,r){const i=iS(n),a=e.value||"",o=i==="`"?"GraveAccent":"Tilde";if(rS(e,n)){const c=n.enter("codeIndented"),f=n.indentLines(a,oS);return c(),f}const s=n.createTracker(r),l=i.repeat(Math.max(nS(a,i)+1,3)),u=n.enter("codeFenced");let d=s.move(l);if(e.lang){const c=n.enter(`codeFencedLang${o}`);d+=s.move(n.safe(e.lang,{before:d,after:" ",encode:["`"],...s.current()})),c()}if(e.lang&&e.meta){const c=n.enter(`codeFencedMeta${o}`);d+=s.move(" "),d+=s.move(n.safe(e.meta,{before:d,after:`
`,encode:["`"],...s.current()})),c()}return d+=s.move(`
`),a&&(d+=s.move(a+`
`)),d+=s.move(l),u(),d}function oS(e,t,n){return(n?"":"    ")+e}function Ec(e){const t=e.options.quote||'"';if(t!=='"'&&t!=="'")throw new Error("Cannot serialize title with `"+t+"` for `options.quote`, expected `\"`, or `'`");return t}function sS(e,t,n,r){const i=Ec(n),a=i==='"'?"Quote":"Apostrophe",o=n.enter("definition");let s=n.enter("label");const l=n.createTracker(r);let u=l.move("[");return u+=l.move(n.safe(n.associationId(e),{before:u,after:"]",...l.current()})),u+=l.move("]: "),s(),!e.url||/[\0- \u007F]/.test(e.url)?(s=n.enter("destinationLiteral"),u+=l.move("<"),u+=l.move(n.safe(e.url,{before:u,after:">",...l.current()})),u+=l.move(">")):(s=n.enter("destinationRaw"),u+=l.move(n.safe(e.url,{before:u,after:e.title?" ":`
`,...l.current()}))),s(),e.title&&(s=n.enter(`title${a}`),u+=l.move(" "+i),u+=l.move(n.safe(e.title,{before:u,after:i,...l.current()})),u+=l.move(i),s()),o(),u}function lS(e){const t=e.options.emphasis||"*";if(t!=="*"&&t!=="_")throw new Error("Cannot serialize emphasis with `"+t+"` for `options.emphasis`, expected `*`, or `_`");return t}function Qi(e){return"&#x"+e.toString(16).toUpperCase()+";"}function zo(e,t,n){const r=qr(e),i=qr(t);return r===void 0?i===void 0?n==="_"?{inside:!0,outside:!0}:{inside:!1,outside:!1}:i===1?{inside:!0,outside:!0}:{inside:!1,outside:!0}:r===1?i===void 0?{inside:!1,outside:!1}:i===1?{inside:!0,outside:!0}:{inside:!1,outside:!1}:i===void 0?{inside:!1,outside:!1}:i===1?{inside:!0,outside:!1}:{inside:!1,outside:!1}}Tg.peek=uS;function Tg(e,t,n,r){const i=lS(n),a=n.enter("emphasis"),o=n.createTracker(r),s=o.move(i);let l=o.move(n.containerPhrasing(e,{after:i,before:s,...o.current()}));const u=l.charCodeAt(0),d=zo(r.before.charCodeAt(r.before.length-1),u,i);d.inside&&(l=Qi(u)+l.slice(1));const c=l.charCodeAt(l.length-1),f=zo(r.after.charCodeAt(0),c,i);f.inside&&(l=l.slice(0,-1)+Qi(c));const m=o.move(i);return a(),n.attentionEncodeSurroundingInfo={after:f.outside,before:d.outside},s+l+m}function uS(e,t,n){return n.options.emphasis||"*"}function cS(e,t){let n=!1;return Sc(e,function(r){if("value"in r&&/\r?\n|\r/.test(r.value)||r.type==="break")return n=!0,su}),!!((!e.depth||e.depth<3)&&yc(e)&&(t.options.setext||n))}function dS(e,t,n,r){const i=Math.max(Math.min(6,e.depth||1),1),a=n.createTracker(r);if(cS(e,n)){const d=n.enter("headingSetext"),c=n.enter("phrasing"),f=n.containerPhrasing(e,{...a.current(),before:`
`,after:`
`});return c(),d(),f+`
`+(i===1?"=":"-").repeat(f.length-(Math.max(f.lastIndexOf("\r"),f.lastIndexOf(`
`))+1))}const o="#".repeat(i),s=n.enter("headingAtx"),l=n.enter("phrasing");a.move(o+" ");let u=n.containerPhrasing(e,{before:"# ",after:`
`,...a.current()});return/^[\t ]/.test(u)&&(u=Qi(u.charCodeAt(0))+u.slice(1)),u=u?o+" "+u:o,n.options.closeAtx&&(u+=" "+o),l(),s(),u}Pg.peek=pS;function Pg(e){return e.value||""}function pS(){return"<"}Ng.peek=fS;function Ng(e,t,n,r){const i=Ec(n),a=i==='"'?"Quote":"Apostrophe",o=n.enter("image");let s=n.enter("label");const l=n.createTracker(r);let u=l.move("![");return u+=l.move(n.safe(e.alt,{before:u,after:"]",...l.current()})),u+=l.move("]("),s(),!e.url&&e.title||/[\0- \u007F]/.test(e.url)?(s=n.enter("destinationLiteral"),u+=l.move("<"),u+=l.move(n.safe(e.url,{before:u,after:">",...l.current()})),u+=l.move(">")):(s=n.enter("destinationRaw"),u+=l.move(n.safe(e.url,{before:u,after:e.title?" ":")",...l.current()}))),s(),e.title&&(s=n.enter(`title${a}`),u+=l.move(" "+i),u+=l.move(n.safe(e.title,{before:u,after:i,...l.current()})),u+=l.move(i),s()),u+=l.move(")"),o(),u}function fS(){return"!"}jg.peek=mS;function jg(e,t,n,r){const i=e.referenceType,a=n.enter("imageReference");let o=n.enter("label");const s=n.createTracker(r);let l=s.move("![");const u=n.safe(e.alt,{before:l,after:"]",...s.current()});l+=s.move(u+"]["),o();const d=n.stack;n.stack=[],o=n.enter("reference");const c=n.safe(n.associationId(e),{before:l,after:"]",...s.current()});return o(),n.stack=d,a(),i==="full"||!u||u!==c?l+=s.move(c+"]"):i==="shortcut"?l=l.slice(0,-1):l+=s.move("]"),l}function mS(){return"!"}Lg.peek=hS;function Lg(e,t,n){let r=e.value||"",i="`",a=-1;for(;new RegExp("(^|[^`])"+i+"([^`]|$)").test(r);)i+="`";for(/[^ \r\n]/.test(r)&&(/^[ \r\n]/.test(r)&&/[ \r\n]$/.test(r)||/^`|`$/.test(r))&&(r=" "+r+" ");++a<n.unsafe.length;){const o=n.unsafe[a],s=n.compilePattern(o);let l;if(o.atBreak)for(;l=s.exec(r);){let u=l.index;r.charCodeAt(u)===10&&r.charCodeAt(u-1)===13&&u--,r=r.slice(0,u)+" "+r.slice(l.index+1)}}return i+r+i}function hS(){return"`"}function Ag(e,t){const n=yc(e);return!!(!t.options.resourceLink&&e.url&&!e.title&&e.children&&e.children.length===1&&e.children[0].type==="text"&&(n===e.url||"mailto:"+n===e.url)&&/^[a-z][a-z+.-]+:/i.test(e.url)&&!/[\0- <>\u007F]/.test(e.url))}Rg.peek=gS;function Rg(e,t,n,r){const i=Ec(n),a=i==='"'?"Quote":"Apostrophe",o=n.createTracker(r);let s,l;if(Ag(e,n)){const d=n.stack;n.stack=[],s=n.enter("autolink");let c=o.move("<");return c+=o.move(n.containerPhrasing(e,{before:c,after:">",...o.current()})),c+=o.move(">"),s(),n.stack=d,c}s=n.enter("link"),l=n.enter("label");let u=o.move("[");return u+=o.move(n.containerPhrasing(e,{before:u,after:"](",...o.current()})),u+=o.move("]("),l(),!e.url&&e.title||/[\0- \u007F]/.test(e.url)?(l=n.enter("destinationLiteral"),u+=o.move("<"),u+=o.move(n.safe(e.url,{before:u,after:">",...o.current()})),u+=o.move(">")):(l=n.enter("destinationRaw"),u+=o.move(n.safe(e.url,{before:u,after:e.title?" ":")",...o.current()}))),l(),e.title&&(l=n.enter(`title${a}`),u+=o.move(" "+i),u+=o.move(n.safe(e.title,{before:u,after:i,...o.current()})),u+=o.move(i),l()),u+=o.move(")"),s(),u}function gS(e,t,n){return Ag(e,n)?"<":"["}Ig.peek=yS;function Ig(e,t,n,r){const i=e.referenceType,a=n.enter("linkReference");let o=n.enter("label");const s=n.createTracker(r);let l=s.move("[");const u=n.containerPhrasing(e,{before:l,after:"]",...s.current()});l+=s.move(u+"]["),o();const d=n.stack;n.stack=[],o=n.enter("reference");const c=n.safe(n.associationId(e),{before:l,after:"]",...s.current()});return o(),n.stack=d,a(),i==="full"||!u||u!==c?l+=s.move(c+"]"):i==="shortcut"?l=l.slice(0,-1):l+=s.move("]"),l}function yS(){return"["}function zc(e){const t=e.options.bullet||"*";if(t!=="*"&&t!=="+"&&t!=="-")throw new Error("Cannot serialize items with `"+t+"` for `options.bullet`, expected `*`, `+`, or `-`");return t}function xS(e){const t=zc(e),n=e.options.bulletOther;if(!n)return t==="*"?"-":"*";if(n!=="*"&&n!=="+"&&n!=="-")throw new Error("Cannot serialize items with `"+n+"` for `options.bulletOther`, expected `*`, `+`, or `-`");if(n===t)throw new Error("Expected `bullet` (`"+t+"`) and `bulletOther` (`"+n+"`) to be different");return n}function _S(e){const t=e.options.bulletOrdered||".";if(t!=="."&&t!==")")throw new Error("Cannot serialize items with `"+t+"` for `options.bulletOrdered`, expected `.` or `)`");return t}function Og(e){const t=e.options.rule||"*";if(t!=="*"&&t!=="-"&&t!=="_")throw new Error("Cannot serialize rules with `"+t+"` for `options.rule`, expected `*`, `-`, or `_`");return t}function vS(e,t,n,r){const i=n.enter("list"),a=n.bulletCurrent;let o=e.ordered?_S(n):zc(n);const s=e.ordered?o==="."?")":".":xS(n);let l=t&&n.bulletLastUsed?o===n.bulletLastUsed:!1;if(!e.ordered){const d=e.children?e.children[0]:void 0;if((o==="*"||o==="-")&&d&&(!d.children||!d.children[0])&&n.stack[n.stack.length-1]==="list"&&n.stack[n.stack.length-2]==="listItem"&&n.stack[n.stack.length-3]==="list"&&n.stack[n.stack.length-4]==="listItem"&&n.indexStack[n.indexStack.length-1]===0&&n.indexStack[n.indexStack.length-2]===0&&n.indexStack[n.indexStack.length-3]===0&&(l=!0),Og(n)===o&&d){let c=-1;for(;++c<e.children.length;){const f=e.children[c];if(f&&f.type==="listItem"&&f.children&&f.children[0]&&f.children[0].type==="thematicBreak"){l=!0;break}}}}l&&(o=s),n.bulletCurrent=o;const u=n.containerFlow(e,r);return n.bulletLastUsed=o,n.bulletCurrent=a,i(),u}function bS(e){const t=e.options.listItemIndent||"one";if(t!=="tab"&&t!=="one"&&t!=="mixed")throw new Error("Cannot serialize items with `"+t+"` for `options.listItemIndent`, expected `tab`, `one`, or `mixed`");return t}function wS(e,t,n,r){const i=bS(n);let a=n.bulletCurrent||zc(n);t&&t.type==="list"&&t.ordered&&(a=(typeof t.start=="number"&&t.start>-1?t.start:1)+(n.options.incrementListMarker===!1?0:t.children.indexOf(e))+a);let o=a.length+1;(i==="tab"||i==="mixed"&&(t&&t.type==="list"&&t.spread||e.spread))&&(o=Math.ceil(o/4)*4);const s=n.createTracker(r);s.move(a+" ".repeat(o-a.length)),s.shift(o);const l=n.enter("listItem"),u=n.indentLines(n.containerFlow(e,s.current()),d);return l(),u;function d(c,f,m){return f?(m?"":" ".repeat(o))+c:(m?a:a+" ".repeat(o-a.length))+c}}function kS(e,t,n,r){const i=n.enter("paragraph"),a=n.enter("phrasing"),o=n.containerPhrasing(e,r);return a(),i(),o}const SS=Zo(["break","delete","emphasis","footnote","footnoteReference","image","imageReference","inlineCode","inlineMath","link","linkReference","mdxJsxTextElement","mdxTextExpression","strong","text","textDirective"]);function CS(e,t,n,r){return(e.children.some(function(o){return SS(o)})?n.containerPhrasing:n.containerFlow).call(n,e,r)}function ES(e){const t=e.options.strong||"*";if(t!=="*"&&t!=="_")throw new Error("Cannot serialize strong with `"+t+"` for `options.strong`, expected `*`, or `_`");return t}Dg.peek=zS;function Dg(e,t,n,r){const i=ES(n),a=n.enter("strong"),o=n.createTracker(r),s=o.move(i+i);let l=o.move(n.containerPhrasing(e,{after:i,before:s,...o.current()}));const u=l.charCodeAt(0),d=zo(r.before.charCodeAt(r.before.length-1),u,i);d.inside&&(l=Qi(u)+l.slice(1));const c=l.charCodeAt(l.length-1),f=zo(r.after.charCodeAt(0),c,i);f.inside&&(l=l.slice(0,-1)+Qi(c));const m=o.move(i+i);return a(),n.attentionEncodeSurroundingInfo={after:f.outside,before:d.outside},s+l+m}function zS(e,t,n){return n.options.strong||"*"}function TS(e,t,n,r){return n.safe(e.value,r)}function PS(e){const t=e.options.ruleRepetition||3;if(t<3)throw new Error("Cannot serialize rules with repetition `"+t+"` for `options.ruleRepetition`, expected `3` or more");return t}function NS(e,t,n){const r=(Og(n)+(n.options.ruleSpaces?" ":"")).repeat(PS(n));return n.options.ruleSpaces?r.slice(0,-1):r}const Fg={blockquote:J2,break:$p,code:aS,definition:sS,emphasis:Tg,hardBreak:$p,heading:dS,html:Pg,image:Ng,imageReference:jg,inlineCode:Lg,link:Rg,linkReference:Ig,list:vS,listItem:wS,paragraph:kS,root:CS,strong:Dg,text:TS,thematicBreak:NS};function jS(){return{enter:{table:LS,tableData:Qp,tableHeader:Qp,tableRow:RS},exit:{codeText:IS,table:AS,tableData:$s,tableHeader:$s,tableRow:$s}}}function LS(e){const t=e._align;this.enter({type:"table",align:t.map(function(n){return n==="none"?null:n}),children:[]},e),this.data.inTable=!0}function AS(e){this.exit(e),this.data.inTable=void 0}function RS(e){this.enter({type:"tableRow",children:[]},e)}function $s(e){this.exit(e)}function Qp(e){this.enter({type:"tableCell",children:[]},e)}function IS(e){let t=this.resume();this.data.inTable&&(t=t.replace(/\\([\\|])/g,OS));const n=this.stack[this.stack.length-1];n.type,n.value=t,this.exit(e)}function OS(e,t){return t==="|"?t:e}function DS(e){const t=e||{},n=t.tableCellPadding,r=t.tablePipeAlign,i=t.stringLength,a=n?" ":"|";return{unsafe:[{character:"\r",inConstruct:"tableCell"},{character:`
`,inConstruct:"tableCell"},{atBreak:!0,character:"|",after:"[	 :-]"},{character:"|",inConstruct:"tableCell"},{atBreak:!0,character:":",after:"-"},{atBreak:!0,character:"-",after:"[:|-]"}],handlers:{inlineCode:f,table:o,tableCell:l,tableRow:s}};function o(m,y,_,k){return u(d(m,_,k),m.align)}function s(m,y,_,k){const h=c(m,_,k),g=u([h]);return g.slice(0,g.indexOf(`
`))}function l(m,y,_,k){const h=_.enter("tableCell"),g=_.enter("phrasing"),x=_.containerPhrasing(m,{...k,before:a,after:a});return g(),h(),x}function u(m,y){return G2(m,{align:y,alignDelimiters:r,padding:n,stringLength:i})}function d(m,y,_){const k=m.children;let h=-1;const g=[],x=y.enter("table");for(;++h<k.length;)g[h]=c(k[h],y,_);return x(),g}function c(m,y,_){const k=m.children;let h=-1;const g=[],x=y.enter("tableRow");for(;++h<k.length;)g[h]=l(k[h],m,y,_);return x(),g}function f(m,y,_){let k=Fg.inlineCode(m,y,_);return _.stack.includes("tableCell")&&(k=k.replace(/\|/g,"\\$&")),k}}function FS(){return{exit:{taskListCheckValueChecked:Kp,taskListCheckValueUnchecked:Kp,paragraph:qS}}}function MS(){return{unsafe:[{atBreak:!0,character:"-",after:"[:|-]"}],handlers:{listItem:WS}}}function Kp(e){const t=this.stack[this.stack.length-2];t.type,t.checked=e.type==="taskListCheckValueChecked"}function qS(e){const t=this.stack[this.stack.length-2];if(t&&t.type==="listItem"&&typeof t.checked=="boolean"){const n=this.stack[this.stack.length-1];n.type;const r=n.children[0];if(r&&r.type==="text"){const i=t.children;let a=-1,o;for(;++a<i.length;){const s=i[a];if(s.type==="paragraph"){o=s;break}}o===n&&(r.value=r.value.slice(1),r.value.length===0?n.children.shift():n.position&&r.position&&typeof r.position.start.offset=="number"&&(r.position.start.column++,r.position.start.offset++,n.position.start=Object.assign({},r.position.start)))}}this.exit(e)}function WS(e,t,n,r){const i=e.children[0],a=typeof e.checked=="boolean"&&i&&i.type==="paragraph",o="["+(e.checked?"x":" ")+"] ",s=n.createTracker(r);a&&s.move(o);let l=Fg.listItem(e,t,n,{...r,...s.current()});return a&&(l=l.replace(/^(?:[*+-]|\d+\.)([\r\n]| {1,3})/,u)),l;function u(d){return d+o}}function BS(){return[v2(),W2(),H2(),jS(),FS()]}function VS(e){return{extensions:[b2(),B2(e),$2(),DS(e),MS()]}}const US={tokenize:XS,partial:!0},Mg={tokenize:GS,partial:!0},qg={tokenize:ZS,partial:!0},Wg={tokenize:JS,partial:!0},HS={tokenize:eC,partial:!0},Bg={name:"wwwAutolink",tokenize:KS,previous:Ug},Vg={name:"protocolAutolink",tokenize:YS,previous:Hg},an={name:"emailAutolink",tokenize:QS,previous:$g},Vt={};function $S(){return{text:Vt}}let On=48;for(;On<123;)Vt[On]=an,On++,On===58?On=65:On===91&&(On=97);Vt[43]=an;Vt[45]=an;Vt[46]=an;Vt[95]=an;Vt[72]=[an,Vg];Vt[104]=[an,Vg];Vt[87]=[an,Bg];Vt[119]=[an,Bg];function QS(e,t,n){const r=this;let i,a;return o;function o(c){return!du(c)||!$g.call(r,r.previous)||Tc(r.events)?n(c):(e.enter("literalAutolink"),e.enter("literalAutolinkEmail"),s(c))}function s(c){return du(c)?(e.consume(c),s):c===64?(e.consume(c),l):n(c)}function l(c){return c===46?e.check(HS,d,u)(c):c===45||c===95||Me(c)?(a=!0,e.consume(c),l):d(c)}function u(c){return e.consume(c),i=!0,l}function d(c){return a&&i&&Ue(r.previous)?(e.exit("literalAutolinkEmail"),e.exit("literalAutolink"),t(c)):n(c)}}function KS(e,t,n){const r=this;return i;function i(o){return o!==87&&o!==119||!Ug.call(r,r.previous)||Tc(r.events)?n(o):(e.enter("literalAutolink"),e.enter("literalAutolinkWww"),e.check(US,e.attempt(Mg,e.attempt(qg,a),n),n)(o))}function a(o){return e.exit("literalAutolinkWww"),e.exit("literalAutolink"),t(o)}}function YS(e,t,n){const r=this;let i="",a=!1;return o;function o(c){return(c===72||c===104)&&Hg.call(r,r.previous)&&!Tc(r.events)?(e.enter("literalAutolink"),e.enter("literalAutolinkHttp"),i+=String.fromCodePoint(c),e.consume(c),s):n(c)}function s(c){if(Ue(c)&&i.length<5)return i+=String.fromCodePoint(c),e.consume(c),s;if(c===58){const f=i.toLowerCase();if(f==="http"||f==="https")return e.consume(c),l}return n(c)}function l(c){return c===47?(e.consume(c),a?u:(a=!0,l)):n(c)}function u(c){return c===null||So(c)||ce(c)||Zn(c)||Yo(c)?n(c):e.attempt(Mg,e.attempt(qg,d),n)(c)}function d(c){return e.exit("literalAutolinkHttp"),e.exit("literalAutolink"),t(c)}}function XS(e,t,n){let r=0;return i;function i(o){return(o===87||o===119)&&r<3?(r++,e.consume(o),i):o===46&&r===3?(e.consume(o),a):n(o)}function a(o){return o===null?n(o):t(o)}}function GS(e,t,n){let r,i,a;return o;function o(u){return u===46||u===95?e.check(Wg,l,s)(u):u===null||ce(u)||Zn(u)||u!==45&&Yo(u)?l(u):(a=!0,e.consume(u),o)}function s(u){return u===95?r=!0:(i=r,r=void 0),e.consume(u),o}function l(u){return i||r||!a?n(u):t(u)}}function ZS(e,t){let n=0,r=0;return i;function i(o){return o===40?(n++,e.consume(o),i):o===41&&r<n?a(o):o===33||o===34||o===38||o===39||o===41||o===42||o===44||o===46||o===58||o===59||o===60||o===63||o===93||o===95||o===126?e.check(Wg,t,a)(o):o===null||ce(o)||Zn(o)?t(o):(e.consume(o),i)}function a(o){return o===41&&r++,e.consume(o),i}}function JS(e,t,n){return r;function r(s){return s===33||s===34||s===39||s===41||s===42||s===44||s===46||s===58||s===59||s===63||s===95||s===126?(e.consume(s),r):s===38?(e.consume(s),a):s===93?(e.consume(s),i):s===60||s===null||ce(s)||Zn(s)?t(s):n(s)}function i(s){return s===null||s===40||s===91||ce(s)||Zn(s)?t(s):r(s)}function a(s){return Ue(s)?o(s):n(s)}function o(s){return s===59?(e.consume(s),r):Ue(s)?(e.consume(s),o):n(s)}}function eC(e,t,n){return r;function r(a){return e.consume(a),i}function i(a){return Me(a)?n(a):t(a)}}function Ug(e){return e===null||e===40||e===42||e===95||e===91||e===93||e===126||ce(e)}function Hg(e){return!Ue(e)}function $g(e){return!(e===47||du(e))}function du(e){return e===43||e===45||e===46||e===95||Me(e)}function Tc(e){let t=e.length,n=!1;for(;t--;){const r=e[t][1];if((r.type==="labelLink"||r.type==="labelImage")&&!r._balanced){n=!0;break}if(r._gfmAutolinkLiteralWalkedInto){n=!1;break}}return e.length>0&&!n&&(e[e.length-1][1]._gfmAutolinkLiteralWalkedInto=!0),n}const tC={tokenize:uC,partial:!0};function nC(){return{document:{91:{name:"gfmFootnoteDefinition",tokenize:oC,continuation:{tokenize:sC},exit:lC}},text:{91:{name:"gfmFootnoteCall",tokenize:aC},93:{name:"gfmPotentialFootnoteCall",add:"after",tokenize:rC,resolveTo:iC}}}}function rC(e,t,n){const r=this;let i=r.events.length;const a=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let o;for(;i--;){const l=r.events[i][1];if(l.type==="labelImage"){o=l;break}if(l.type==="gfmFootnoteCall"||l.type==="labelLink"||l.type==="label"||l.type==="image"||l.type==="link")break}return s;function s(l){if(!o||!o._balanced)return n(l);const u=Rt(r.sliceSerialize({start:o.end,end:r.now()}));return u.codePointAt(0)!==94||!a.includes(u.slice(1))?n(l):(e.enter("gfmFootnoteCallLabelMarker"),e.consume(l),e.exit("gfmFootnoteCallLabelMarker"),t(l))}}function iC(e,t){let n=e.length;for(;n--;)if(e[n][1].type==="labelImage"&&e[n][0]==="enter"){e[n][1];break}e[n+1][1].type="data",e[n+3][1].type="gfmFootnoteCallLabelMarker";const r={type:"gfmFootnoteCall",start:Object.assign({},e[n+3][1].start),end:Object.assign({},e[e.length-1][1].end)},i={type:"gfmFootnoteCallMarker",start:Object.assign({},e[n+3][1].end),end:Object.assign({},e[n+3][1].end)};i.end.column++,i.end.offset++,i.end._bufferIndex++;const a={type:"gfmFootnoteCallString",start:Object.assign({},i.end),end:Object.assign({},e[e.length-1][1].start)},o={type:"chunkString",contentType:"string",start:Object.assign({},a.start),end:Object.assign({},a.end)},s=[e[n+1],e[n+2],["enter",r,t],e[n+3],e[n+4],["enter",i,t],["exit",i,t],["enter",a,t],["enter",o,t],["exit",o,t],["exit",a,t],e[e.length-2],e[e.length-1],["exit",r,t]];return e.splice(n,e.length-n+1,...s),e}function aC(e,t,n){const r=this,i=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let a=0,o;return s;function s(c){return e.enter("gfmFootnoteCall"),e.enter("gfmFootnoteCallLabelMarker"),e.consume(c),e.exit("gfmFootnoteCallLabelMarker"),l}function l(c){return c!==94?n(c):(e.enter("gfmFootnoteCallMarker"),e.consume(c),e.exit("gfmFootnoteCallMarker"),e.enter("gfmFootnoteCallString"),e.enter("chunkString").contentType="string",u)}function u(c){if(a>999||c===93&&!o||c===null||c===91||ce(c))return n(c);if(c===93){e.exit("chunkString");const f=e.exit("gfmFootnoteCallString");return i.includes(Rt(r.sliceSerialize(f)))?(e.enter("gfmFootnoteCallLabelMarker"),e.consume(c),e.exit("gfmFootnoteCallLabelMarker"),e.exit("gfmFootnoteCall"),t):n(c)}return ce(c)||(o=!0),a++,e.consume(c),c===92?d:u}function d(c){return c===91||c===92||c===93?(e.consume(c),a++,u):u(c)}}function oC(e,t,n){const r=this,i=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let a,o=0,s;return l;function l(y){return e.enter("gfmFootnoteDefinition")._container=!0,e.enter("gfmFootnoteDefinitionLabel"),e.enter("gfmFootnoteDefinitionLabelMarker"),e.consume(y),e.exit("gfmFootnoteDefinitionLabelMarker"),u}function u(y){return y===94?(e.enter("gfmFootnoteDefinitionMarker"),e.consume(y),e.exit("gfmFootnoteDefinitionMarker"),e.enter("gfmFootnoteDefinitionLabelString"),e.enter("chunkString").contentType="string",d):n(y)}function d(y){if(o>999||y===93&&!s||y===null||y===91||ce(y))return n(y);if(y===93){e.exit("chunkString");const _=e.exit("gfmFootnoteDefinitionLabelString");return a=Rt(r.sliceSerialize(_)),e.enter("gfmFootnoteDefinitionLabelMarker"),e.consume(y),e.exit("gfmFootnoteDefinitionLabelMarker"),e.exit("gfmFootnoteDefinitionLabel"),f}return ce(y)||(s=!0),o++,e.consume(y),y===92?c:d}function c(y){return y===91||y===92||y===93?(e.consume(y),o++,d):d(y)}function f(y){return y===58?(e.enter("definitionMarker"),e.consume(y),e.exit("definitionMarker"),i.includes(a)||i.push(a),re(e,m,"gfmFootnoteDefinitionWhitespace")):n(y)}function m(y){return t(y)}}function sC(e,t,n){return e.check(aa,t,e.attempt(tC,t,n))}function lC(e){e.exit("gfmFootnoteDefinition")}function uC(e,t,n){const r=this;return re(e,i,"gfmFootnoteDefinitionIndent",5);function i(a){const o=r.events[r.events.length-1];return o&&o[1].type==="gfmFootnoteDefinitionIndent"&&o[2].sliceSerialize(o[1],!0).length===4?t(a):n(a)}}function cC(e){let n=(e||{}).singleTilde;const r={name:"strikethrough",tokenize:a,resolveAll:i};return n==null&&(n=!0),{text:{126:r},insideSpan:{null:[r]},attentionMarkers:{null:[126]}};function i(o,s){let l=-1;for(;++l<o.length;)if(o[l][0]==="enter"&&o[l][1].type==="strikethroughSequenceTemporary"&&o[l][1]._close){let u=l;for(;u--;)if(o[u][0]==="exit"&&o[u][1].type==="strikethroughSequenceTemporary"&&o[u][1]._open&&o[l][1].end.offset-o[l][1].start.offset===o[u][1].end.offset-o[u][1].start.offset){o[l][1].type="strikethroughSequence",o[u][1].type="strikethroughSequence";const d={type:"strikethrough",start:Object.assign({},o[u][1].start),end:Object.assign({},o[l][1].end)},c={type:"strikethroughText",start:Object.assign({},o[u][1].end),end:Object.assign({},o[l][1].start)},f=[["enter",d,s],["enter",o[u][1],s],["exit",o[u][1],s],["enter",c,s]],m=s.parser.constructs.insideSpan.null;m&&lt(f,f.length,0,Xo(m,o.slice(u+1,l),s)),lt(f,f.length,0,[["exit",c,s],["enter",o[l][1],s],["exit",o[l][1],s],["exit",d,s]]),lt(o,u-1,l-u+3,f),l=u+f.length-2;break}}for(l=-1;++l<o.length;)o[l][1].type==="strikethroughSequenceTemporary"&&(o[l][1].type="data");return o}function a(o,s,l){const u=this.previous,d=this.events;let c=0;return f;function f(y){return u===126&&d[d.length-1][1].type!=="characterEscape"?l(y):(o.enter("strikethroughSequenceTemporary"),m(y))}function m(y){const _=qr(u);if(y===126)return c>1?l(y):(o.consume(y),c++,m);if(c<2&&!n)return l(y);const k=o.exit("strikethroughSequenceTemporary"),h=qr(y);return k._open=!h||h===2&&!!_,k._close=!_||_===2&&!!h,s(y)}}}class dC{constructor(){this.map=[]}add(t,n,r){pC(this,t,n,r)}consume(t){if(this.map.sort(function(a,o){return a[0]-o[0]}),this.map.length===0)return;let n=this.map.length;const r=[];for(;n>0;)n-=1,r.push(t.slice(this.map[n][0]+this.map[n][1]),this.map[n][2]),t.length=this.map[n][0];r.push(t.slice()),t.length=0;let i=r.pop();for(;i;){for(const a of i)t.push(a);i=r.pop()}this.map.length=0}}function pC(e,t,n,r){let i=0;if(!(n===0&&r.length===0)){for(;i<e.map.length;){if(e.map[i][0]===t){e.map[i][1]+=n,e.map[i][2].push(...r);return}i+=1}e.map.push([t,n,r])}}function fC(e,t){let n=!1;const r=[];for(;t<e.length;){const i=e[t];if(n){if(i[0]==="enter")i[1].type==="tableContent"&&r.push(e[t+1][1].type==="tableDelimiterMarker"?"left":"none");else if(i[1].type==="tableContent"){if(e[t-1][1].type==="tableDelimiterMarker"){const a=r.length-1;r[a]=r[a]==="left"?"center":"right"}}else if(i[1].type==="tableDelimiterRow")break}else i[0]==="enter"&&i[1].type==="tableDelimiterRow"&&(n=!0);t+=1}return r}function mC(){return{flow:{null:{name:"table",tokenize:hC,resolveAll:gC}}}}function hC(e,t,n){const r=this;let i=0,a=0,o;return s;function s(w){let I=r.events.length-1;for(;I>-1;){const ee=r.events[I][1].type;if(ee==="lineEnding"||ee==="linePrefix")I--;else break}const W=I>-1?r.events[I][1].type:null,Y=W==="tableHead"||W==="tableRow"?S:l;return Y===S&&r.parser.lazy[r.now().line]?n(w):Y(w)}function l(w){return e.enter("tableHead"),e.enter("tableRow"),u(w)}function u(w){return w===124||(o=!0,a+=1),d(w)}function d(w){return w===null?n(w):K(w)?a>1?(a=0,r.interrupt=!0,e.exit("tableRow"),e.enter("lineEnding"),e.consume(w),e.exit("lineEnding"),m):n(w):Z(w)?re(e,d,"whitespace")(w):(a+=1,o&&(o=!1,i+=1),w===124?(e.enter("tableCellDivider"),e.consume(w),e.exit("tableCellDivider"),o=!0,d):(e.enter("data"),c(w)))}function c(w){return w===null||w===124||ce(w)?(e.exit("data"),d(w)):(e.consume(w),w===92?f:c)}function f(w){return w===92||w===124?(e.consume(w),c):c(w)}function m(w){return r.interrupt=!1,r.parser.lazy[r.now().line]?n(w):(e.enter("tableDelimiterRow"),o=!1,Z(w)?re(e,y,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(w):y(w))}function y(w){return w===45||w===58?k(w):w===124?(o=!0,e.enter("tableCellDivider"),e.consume(w),e.exit("tableCellDivider"),_):z(w)}function _(w){return Z(w)?re(e,k,"whitespace")(w):k(w)}function k(w){return w===58?(a+=1,o=!0,e.enter("tableDelimiterMarker"),e.consume(w),e.exit("tableDelimiterMarker"),h):w===45?(a+=1,h(w)):w===null||K(w)?C(w):z(w)}function h(w){return w===45?(e.enter("tableDelimiterFiller"),g(w)):z(w)}function g(w){return w===45?(e.consume(w),g):w===58?(o=!0,e.exit("tableDelimiterFiller"),e.enter("tableDelimiterMarker"),e.consume(w),e.exit("tableDelimiterMarker"),x):(e.exit("tableDelimiterFiller"),x(w))}function x(w){return Z(w)?re(e,C,"whitespace")(w):C(w)}function C(w){return w===124?y(w):w===null||K(w)?!o||i!==a?z(w):(e.exit("tableDelimiterRow"),e.exit("tableHead"),t(w)):z(w)}function z(w){return n(w)}function S(w){return e.enter("tableRow"),T(w)}function T(w){return w===124?(e.enter("tableCellDivider"),e.consume(w),e.exit("tableCellDivider"),T):w===null||K(w)?(e.exit("tableRow"),t(w)):Z(w)?re(e,T,"whitespace")(w):(e.enter("data"),N(w))}function N(w){return w===null||w===124||ce(w)?(e.exit("data"),T(w)):(e.consume(w),w===92?M:N)}function M(w){return w===92||w===124?(e.consume(w),N):N(w)}}function gC(e,t){let n=-1,r=!0,i=0,a=[0,0,0,0],o=[0,0,0,0],s=!1,l=0,u,d,c;const f=new dC;for(;++n<e.length;){const m=e[n],y=m[1];m[0]==="enter"?y.type==="tableHead"?(s=!1,l!==0&&(Yp(f,t,l,u,d),d=void 0,l=0),u={type:"table",start:Object.assign({},y.start),end:Object.assign({},y.end)},f.add(n,0,[["enter",u,t]])):y.type==="tableRow"||y.type==="tableDelimiterRow"?(r=!0,c=void 0,a=[0,0,0,0],o=[0,n+1,0,0],s&&(s=!1,d={type:"tableBody",start:Object.assign({},y.start),end:Object.assign({},y.end)},f.add(n,0,[["enter",d,t]])),i=y.type==="tableDelimiterRow"?2:d?3:1):i&&(y.type==="data"||y.type==="tableDelimiterMarker"||y.type==="tableDelimiterFiller")?(r=!1,o[2]===0&&(a[1]!==0&&(o[0]=o[1],c=Pa(f,t,a,i,void 0,c),a=[0,0,0,0]),o[2]=n)):y.type==="tableCellDivider"&&(r?r=!1:(a[1]!==0&&(o[0]=o[1],c=Pa(f,t,a,i,void 0,c)),a=o,o=[a[1],n,0,0])):y.type==="tableHead"?(s=!0,l=n):y.type==="tableRow"||y.type==="tableDelimiterRow"?(l=n,a[1]!==0?(o[0]=o[1],c=Pa(f,t,a,i,n,c)):o[1]!==0&&(c=Pa(f,t,o,i,n,c)),i=0):i&&(y.type==="data"||y.type==="tableDelimiterMarker"||y.type==="tableDelimiterFiller")&&(o[3]=n)}for(l!==0&&Yp(f,t,l,u,d),f.consume(t.events),n=-1;++n<t.events.length;){const m=t.events[n];m[0]==="enter"&&m[1].type==="table"&&(m[1]._align=fC(t.events,n))}return e}function Pa(e,t,n,r,i,a){const o=r===1?"tableHeader":r===2?"tableDelimiter":"tableData",s="tableContent";n[0]!==0&&(a.end=Object.assign({},sr(t.events,n[0])),e.add(n[0],0,[["exit",a,t]]));const l=sr(t.events,n[1]);if(a={type:o,start:Object.assign({},l),end:Object.assign({},l)},e.add(n[1],0,[["enter",a,t]]),n[2]!==0){const u=sr(t.events,n[2]),d=sr(t.events,n[3]),c={type:s,start:Object.assign({},u),end:Object.assign({},d)};if(e.add(n[2],0,[["enter",c,t]]),r!==2){const f=t.events[n[2]],m=t.events[n[3]];if(f[1].end=Object.assign({},m[1].end),f[1].type="chunkText",f[1].contentType="text",n[3]>n[2]+1){const y=n[2]+1,_=n[3]-n[2]-1;e.add(y,_,[])}}e.add(n[3]+1,0,[["exit",c,t]])}return i!==void 0&&(a.end=Object.assign({},sr(t.events,i)),e.add(i,0,[["exit",a,t]]),a=void 0),a}function Yp(e,t,n,r,i){const a=[],o=sr(t.events,n);i&&(i.end=Object.assign({},o),a.push(["exit",i,t])),r.end=Object.assign({},o),a.push(["exit",r,t]),e.add(n+1,0,a)}function sr(e,t){const n=e[t],r=n[0]==="enter"?"start":"end";return n[1][r]}const yC={name:"tasklistCheck",tokenize:_C};function xC(){return{text:{91:yC}}}function _C(e,t,n){const r=this;return i;function i(l){return r.previous!==null||!r._gfmTasklistFirstContentOfListItem?n(l):(e.enter("taskListCheck"),e.enter("taskListCheckMarker"),e.consume(l),e.exit("taskListCheckMarker"),a)}function a(l){return ce(l)?(e.enter("taskListCheckValueUnchecked"),e.consume(l),e.exit("taskListCheckValueUnchecked"),o):l===88||l===120?(e.enter("taskListCheckValueChecked"),e.consume(l),e.exit("taskListCheckValueChecked"),o):n(l)}function o(l){return l===93?(e.enter("taskListCheckMarker"),e.consume(l),e.exit("taskListCheckMarker"),e.exit("taskListCheck"),s):n(l)}function s(l){return K(l)?t(l):Z(l)?e.check({tokenize:vC},t,n)(l):n(l)}}function vC(e,t,n){return re(e,r,"whitespace");function r(i){return i===null?n(i):t(i)}}function bC(e){return tg([$S(),nC(),cC(e),mC(),xC()])}const wC={};function Qg(e){const t=this,n=e||wC,r=t.data(),i=r.micromarkExtensions||(r.micromarkExtensions=[]),a=r.fromMarkdownExtensions||(r.fromMarkdownExtensions=[]),o=r.toMarkdownExtensions||(r.toMarkdownExtensions=[]);i.push(bC(n)),a.push(BS()),o.push(VS(n))}const kC=[{id:"numpy-array-sum",title:"Array Sum",section:"python-basics",difficulty:"easy",description:`
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
`}],SC=[{id:"normalize-features",title:"Normalize Features",section:"data-preprocessing",difficulty:"easy",description:`
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
`}],CC=[{id:"linear-regression-gd",title:"Linear Regression with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
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
`}],EC=[{id:"kmeans-clustering",title:"K-Means: Assign to Nearest Centroid",section:"unsupervised-learning",difficulty:"medium",description:`
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
`}],zC=[{id:"perceptron",title:"ReLU Activation",section:"deep-learning",difficulty:"easy",description:`
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
`}],TC=[{id:"precision-recall-f1",title:"Precision and Recall",section:"model-evaluation",difficulty:"medium",description:`
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
`}],PC=[{id:"mlp-forward-backward",title:"MLP Forward & Backward Pass",section:"neural-networks",difficulty:"hard",description:"\n## MLP Forward & Backward Pass\n\nImplement both forward and backward passes for a 2-layer MLP (Multi-Layer Perceptron).\n\n### Architecture\n```\nInput (n_features) → Hidden (n_hidden) → Output (n_classes)\n```\n\n### Forward Pass\n```\nZ1 = X @ W1 + b1\nA1 = ReLU(Z1)\nZ2 = A1 @ W2 + b2\noutput = softmax(Z2)\n```\n\nReturn: `(output, cache)` where cache = `(X, Z1, A1, Z2, W1, W2)`\n\n### Backward Pass (Backpropagation)\nUsing the chain rule, gradients flow backwards:\n\n```\nLoss → dZ2 → dW2, db2 → dA1 → dZ1 (ReLU) → dW1, db1\n```\n\n**Key Formulas:**\n- `dZ2 = output - Y_onehot` (softmax + cross-entropy derivative)\n- `dW2 = (1/m) * A1.T @ dZ2`\n- `db2 = (1/m) * sum(dZ2, axis=0)`\n- `dZ1 = (dZ2 @ W2.T) * (Z1 > 0)` (ReLU derivative)\n- `dW1 = (1/m) * X.T @ dZ1`\n- `db1 = (1/m) * sum(dZ1, axis=0)`\n\nReturn: `{'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...}`\n    ",examples:[{input:"X shape (2, 3), hidden_size=4, output_size=2",output:"Forward: probabilities (2, 2), rows sum to 1. Backward: gradients for all weights/biases",explanation:"Forward computes predictions; backward computes gradients for training"}],starterCode:`import numpy as np

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
`}],NC=[{id:"conv2d-forward",title:"2D Convolution",section:"cnn",difficulty:"hard",description:`
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
`}],jC=[{id:"scaled-dot-product-attention",title:"Scaled Dot-Product Attention",section:"transformers",difficulty:"medium",description:`
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
`}],LC=[{id:"vae-reparameterization",title:"VAE Reparameterization Trick",section:"generative-models",difficulty:"medium",description:`
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
`}],AC=[{id:"numpy-array-creation",title:"Array Creation Methods",section:"numpy-fundamentals",difficulty:"easy",description:`
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
`}],RC=[{id:"einsum-basics",title:"Einsum Fundamentals",section:"einsum",difficulty:"easy",description:`
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
`}],IC=[{id:"tensor-creation",title:"Tensor Creation (NumPy Style)",section:"pytorch-basics",difficulty:"easy",description:"\n## Tensor Creation\n\nLearn tensor creation patterns used in PyTorch, implemented with NumPy.\n\n### PyTorch Equivalents\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.tensor([1,2,3])` | `np.array([1,2,3])` |\n| `torch.zeros(3, 4)` | `np.zeros((3, 4))` |\n| `torch.randn(2, 3)` | `np.random.randn(2, 3)` |\n| `torch.arange(10)` | `np.arange(10)` |\n| `torch.linspace(0, 1, 5)` | `np.linspace(0, 1, 5)` |\n\n### Data Types\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.float32` | `np.float32` |\n| `torch.int64` | `np.int64` |\n\n### Task\nCreate tensors matching PyTorch patterns.\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- `'from_list'`: Array from [1,2,3,4] with float32 dtype\n- `'zeros'`: Zero array of shape (3, 4)\n- `'ones'`: Ones array of shape (2, 3)\n- `'randn'`: Random normal array of shape (2, 3)\n- `'arange'`: Array from 0 to 9\n- `'linspace'`: 5 evenly spaced values from 0 to 1\n- `'eye'`: 4x4 identity matrix\n- `'full'`: (2, 3) array filled with 7.0\n    ",examples:[{input:"create_tensors()",output:"{'zeros': shape (3,4), 'ones': shape (2,3), 'randn': shape (2,3), 'arange': [0..9], ...}",explanation:"Dictionary containing tensors created with various methods"}],starterCode:`import numpy as np

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
`}],OC=[{id:"e2e-mlp",title:"E2E: 2-Layer MLP with Backprop",section:"e2e-implementations",difficulty:"hard",description:`
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
`}],DC=[{id:"rl-discounted-return",title:"Discounted Return",section:"reinforcement-learning",difficulty:"easy",description:`
## Discounted Return (Cumulative Reward)

Calculate the discounted return (cumulative reward) from a sequence of rewards.

### Formula
\`\`\`
G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ... + γ^(T-t) * r_T
    = Σ_{k=0}^{T-t} γ^k * r_{t+k}
\`\`\`

Where:
- \`r_t\`: Reward at timestep t
- \`γ\` (gamma): Discount factor (0 ≤ γ ≤ 1)
- \`T\`: Final timestep

### Why Discount?
- Future rewards are uncertain
- Prefer immediate rewards
- γ = 0: Only care about immediate reward
- γ = 1: All rewards equally important
- Typical values: 0.9 to 0.99

### Function Signature
\`\`\`python
def discounted_return(rewards: np.ndarray, gamma: float) -> float:
\`\`\`

Returns the total discounted return from timestep 0.
    `,examples:[{input:"rewards = [1, 1, 1, 1], gamma = 0.9",output:"3.439",explanation:"1 + 0.9*1 + 0.81*1 + 0.729*1 = 3.439"},{input:"rewards = [0, 0, 0, 10], gamma = 0.5",output:"1.25",explanation:"0 + 0 + 0 + 0.125*10 = 1.25"}],starterCode:`import numpy as np

def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    """
    Calculate the discounted cumulative return.

    Args:
        rewards: Array of rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor (0 <= gamma <= 1)

    Returns:
        Total discounted return G_0
    """
    rewards = np.array(rewards)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Constant rewards",input:"([1, 1, 1, 1], 0.9)",expected:"3.439",hidden:!1},{id:"2",description:"Delayed reward",input:"([0, 0, 0, 10], 0.5)",expected:"1.25",hidden:!1},{id:"3",description:"No discount",input:"([1, 2, 3, 4], 1.0)",expected:"10.0",hidden:!1},{id:"4",description:"Only immediate",input:"([5, 10, 15], 0.0)",expected:"5.0",hidden:!0}],hints:["Create an array of discount factors: [γ⁰, γ¹, γ², ...]","Use np.arange to create exponents, then gamma ** exponents","Multiply rewards by discount factors and sum"],solution:`import numpy as np

def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    """
    Calculate the discounted cumulative return.
    """
    rewards = np.array(rewards)
    T = len(rewards)

    # Create discount factors: [gamma^0, gamma^1, gamma^2, ...]
    discounts = gamma ** np.arange(T)

    # Compute discounted return
    G = np.sum(rewards * discounts)

    return round(G, 3)
`},{id:"rl-epsilon-greedy",title:"Epsilon-Greedy Policy",section:"reinforcement-learning",difficulty:"easy",description:`
## Epsilon-Greedy Policy

Implement the epsilon-greedy action selection policy, which balances exploration and exploitation.

### Algorithm
\`\`\`
With probability ε: choose random action (explore)
With probability 1-ε: choose action with highest Q-value (exploit)
\`\`\`

### Why Epsilon-Greedy?
- **Exploration**: Try new actions to discover better strategies
- **Exploitation**: Use known good actions to maximize reward
- ε typically decays over time (start exploring, then exploit)

### Function Signature
\`\`\`python
def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   random_value: float) -> int:
\`\`\`

**Note**: We pass \`random_value\` (a number between 0-1) instead of generating it,
so tests are deterministic. In practice, you'd use \`np.random.random()\`.

### Expected Return
- If \`random_value < epsilon\`: return \`random_action\` (passed as floor of random_value * n_actions for testing)
- Else: return \`argmax(q_values)\`
    `,examples:[{input:"q_values = [1.0, 3.0, 2.0], epsilon = 0.1, random_value = 0.5",output:"1",explanation:"0.5 >= 0.1, so exploit: argmax([1,3,2]) = 1"},{input:"q_values = [1.0, 3.0, 2.0], epsilon = 0.9, random_value = 0.5",output:"1",explanation:"0.5 < 0.9, so explore: floor(0.5 * 3) = 1"}],starterCode:`import numpy as np

def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   random_value: float) -> int:
    """
    Select action using epsilon-greedy policy.

    Args:
        q_values: Q-values for each action [Q(a_0), Q(a_1), ...]
        epsilon: Exploration probability (0 <= epsilon <= 1)
        random_value: Pre-generated random number for deterministic testing

    Returns:
        Selected action index
    """
    q_values = np.array(q_values)
    n_actions = len(q_values)

    # Your code here
    # If random_value < epsilon: explore (return floor(random_value * n_actions))
    # Else: exploit (return argmax of q_values)
    pass
`,testCases:[{id:"1",description:"Exploit (greedy)",input:"([1.0, 3.0, 2.0], 0.1, 0.5)",expected:"1",hidden:!1},{id:"2",description:"Explore",input:"([1.0, 3.0, 2.0], 0.9, 0.5)",expected:"1",hidden:!1},{id:"3",description:"Always exploit (epsilon=0)",input:"([5.0, 2.0, 8.0, 1.0], 0.0, 0.99)",expected:"2",hidden:!1},{id:"4",description:"Explore with low random",input:"([1.0, 2.0, 3.0, 4.0], 0.5, 0.2)",expected:"0",hidden:!0}],hints:["Compare random_value to epsilon to decide explore vs exploit","For exploration, compute floor(random_value * n_actions)","For exploitation, use np.argmax(q_values)"],solution:`import numpy as np

def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   random_value: float) -> int:
    """
    Select action using epsilon-greedy policy.
    """
    q_values = np.array(q_values)
    n_actions = len(q_values)

    if random_value < epsilon:
        # Explore: random action
        return int(random_value * n_actions)
    else:
        # Exploit: greedy action
        return int(np.argmax(q_values))
`},{id:"rl-bellman-value",title:"Bellman Expectation (State Value)",section:"reinforcement-learning",difficulty:"easy",description:`
## Bellman Expectation Equation for State Value

Compute the value of a state given transition probabilities, rewards, and next-state values.

### Bellman Expectation Equation
\`\`\`
V(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
\`\`\`

For a fixed policy, this simplifies to:
\`\`\`
V(s) = Σ_{s'} P(s'|s) * [R(s,s') + γ * V(s')]
\`\`\`

### What It Means
The value of state s equals the expected:
- Immediate reward R
- Plus discounted value of next state V(s')

### Function Signature
\`\`\`python
def bellman_value(transitions: list, gamma: float,
                  next_values: np.ndarray) -> float:
\`\`\`

Where each transition is \`(probability, reward, next_state_index)\`.
    `,examples:[{input:"transitions=[(0.5, 1, 0), (0.5, 2, 1)], gamma=0.9, V=[5, 10]",output:"8.25",explanation:"0.5*(1 + 0.9*5) + 0.5*(2 + 0.9*10) = 0.5*5.5 + 0.5*11 = 8.25"}],starterCode:`import numpy as np

def bellman_value(transitions: list, gamma: float,
                  next_values: np.ndarray) -> float:
    """
    Compute state value using Bellman expectation equation.

    Args:
        transitions: List of (probability, reward, next_state_index) tuples
        gamma: Discount factor
        next_values: Array of values V(s') for each state

    Returns:
        Value V(s) for the current state
    """
    next_values = np.array(next_values)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Two transitions",input:"([(0.5, 1, 0), (0.5, 2, 1)], 0.9, [5, 10])",expected:"8.25",hidden:!1},{id:"2",description:"Deterministic transition",input:"([(1.0, 5, 0)], 0.9, [10])",expected:"14.0",hidden:!1},{id:"3",description:"Three outcomes",input:"([(0.2, 0, 0), (0.5, 1, 1), (0.3, 2, 2)], 0.9, [0, 5, 10])",expected:"6.05",hidden:!0}],hints:["Loop through each transition (p, r, s_next)","For each: add p * (r + gamma * V[s_next]) to the total","This is the expected value over all possible transitions"],solution:`import numpy as np

def bellman_value(transitions: list, gamma: float,
                  next_values: np.ndarray) -> float:
    """
    Compute state value using Bellman expectation equation.
    """
    next_values = np.array(next_values)

    value = 0.0
    for prob, reward, next_state in transitions:
        # V(s) = Σ P(s'|s) * [R + γ * V(s')]
        value += prob * (reward + gamma * next_values[next_state])

    return round(value, 2)
`},{id:"rl-q-learning-update",title:"Q-Learning Update",section:"reinforcement-learning",difficulty:"medium",description:`
## Q-Learning Update Rule

Implement the Q-learning update rule, the core of this model-free RL algorithm.

### Q-Learning Update
\`\`\`
Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
\`\`\`

Where:
- \`α\`: Learning rate
- \`r\`: Reward received
- \`γ\`: Discount factor
- \`s'\`: Next state
- \`max_a' Q(s', a')\`: Best Q-value in next state

### Key Properties
- **Off-policy**: Updates use max Q-value, not the action actually taken
- **Model-free**: Doesn't need environment dynamics
- **Converges**: To optimal Q* under certain conditions

### Function Signature
\`\`\`python
def q_learning_update(Q: np.ndarray, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> np.ndarray:
\`\`\`

Returns the updated Q-table.
    `,examples:[{input:"Q=[[0,0],[0,0]], s=0, a=1, r=1, s'=1, α=0.1, γ=0.9",output:"Q[0,1] = 0.1",explanation:"Q[0,1] += 0.1 * (1 + 0.9*0 - 0) = 0.1"}],starterCode:`import numpy as np

def q_learning_update(Q: np.ndarray, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one Q-learning update.

    Args:
        Q: Q-table of shape (n_states, n_actions)
        state: Current state index
        action: Action taken
        reward: Reward received
        next_state: Next state index
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q-table
    """
    Q = np.array(Q, dtype=float)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple update",input:"([[0, 0], [0, 0]], 0, 1, 1.0, 1, 0.1, 0.9)",expected:"[[0.0, 0.1], [0.0, 0.0]]",hidden:!1},{id:"2",description:"With existing values",input:"([[1, 2], [3, 4]], 0, 0, 5.0, 1, 0.5, 0.9)",expected:"[[4.8, 2.0], [3.0, 4.0]]",hidden:!1},{id:"3",description:"Terminal state (no future)",input:"([[0, 0], [0, 0]], 0, 0, 10.0, 1, 1.0, 0.0)",expected:"[[10.0, 0.0], [0.0, 0.0]]",hidden:!0}],hints:["Find the maximum Q-value in the next state: np.max(Q[next_state])","Compute TD target: reward + gamma * max_next_Q","Compute TD error: target - Q[state, action]","Update: Q[state, action] += alpha * td_error"],solution:`import numpy as np

def q_learning_update(Q: np.ndarray, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one Q-learning update.
    """
    Q = np.array(Q, dtype=float)

    # Best Q-value in next state
    max_next_Q = np.max(Q[next_state])

    # TD target (Bellman optimality target)
    td_target = reward + gamma * max_next_Q

    # TD error
    td_error = td_target - Q[state, action]

    # Update Q-value
    Q[state, action] += alpha * td_error

    return np.round(Q, 1).tolist()
`},{id:"rl-sarsa-update",title:"SARSA Update",section:"reinforcement-learning",difficulty:"medium",description:`
## SARSA Update Rule

Implement SARSA (State-Action-Reward-State-Action), an on-policy TD control algorithm.

### SARSA Update
\`\`\`
Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
\`\`\`

### SARSA vs Q-Learning
| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| Policy | On-policy | Off-policy |
| Next Q | Q(s', a') actual | max_a' Q(s', a') |
| Safety | More conservative | Can be risky |

### On-Policy Meaning
SARSA uses the **actual next action** a' that will be taken (following the current policy),
while Q-learning uses the **best possible** next action.

### Function Signature
\`\`\`python
def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> np.ndarray:
\`\`\`
    `,examples:[{input:"Q=[[1,2],[3,4]], s=0, a=0, r=5, s'=1, a'=0, α=0.5, γ=0.9",output:"Q[0,0] = 4.35",explanation:"Q[0,0] += 0.5*(5 + 0.9*3 - 1) = 0.5*6.7 = 3.35, so 1+3.35=4.35"}],starterCode:`import numpy as np

def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one SARSA update.

    Args:
        Q: Q-table of shape (n_states, n_actions)
        state: Current state s
        action: Action taken a
        reward: Reward received r
        next_state: Next state s'
        next_action: Next action a' (will be taken)
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q-table
    """
    Q = np.array(Q, dtype=float)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic SARSA update",input:"([[1, 2], [3, 4]], 0, 0, 5.0, 1, 0, 0.5, 0.9)",expected:"[[4.35, 2.0], [3.0, 4.0]]",hidden:!1},{id:"2",description:"Different next action",input:"([[1, 2], [3, 4]], 0, 0, 5.0, 1, 1, 0.5, 0.9)",expected:"[[4.8, 2.0], [3.0, 4.0]]",hidden:!1},{id:"3",description:"From zero",input:"([[0, 0], [0, 0]], 0, 1, 1.0, 1, 0, 0.1, 0.9)",expected:"[[0.0, 0.1], [0.0, 0.0]]",hidden:!0}],hints:["The key difference from Q-learning: use Q[next_state, next_action] not max","TD target: reward + gamma * Q[next_state, next_action]","TD error: target - Q[state, action]","Update: Q[state, action] += alpha * td_error"],solution:`import numpy as np

def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one SARSA update.
    """
    Q = np.array(Q, dtype=float)

    # SARSA uses actual next action, not max
    next_Q = Q[next_state, next_action]

    # TD target
    td_target = reward + gamma * next_Q

    # TD error
    td_error = td_target - Q[state, action]

    # Update Q-value
    Q[state, action] += alpha * td_error

    return np.round(Q, 2).tolist()
`},{id:"rl-td0-prediction",title:"TD(0) Value Prediction",section:"reinforcement-learning",difficulty:"medium",description:`
## TD(0) Value Prediction

Implement TD(0), the simplest temporal difference learning algorithm for value estimation.

### TD(0) Update
\`\`\`
V(s) ← V(s) + α * [r + γ * V(s') - V(s)]
\`\`\`

### Key Concepts
- **TD Error (δ)**: \`r + γ * V(s') - V(s)\`
- **Bootstrap**: Uses estimated V(s') rather than waiting for episode end
- **Online**: Updates after each step, not at episode end

### TD vs Monte Carlo
| Aspect | TD(0) | Monte Carlo |
|--------|-------|-------------|
| Update | Every step | End of episode |
| Bias | Some bias | Unbiased |
| Variance | Low variance | High variance |
| Bootstrap | Yes | No |

### Function Signature
\`\`\`python
def td0_update(V: np.ndarray, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> np.ndarray:
\`\`\`

If \`done=True\`, the next state is terminal (V(s')=0).
    `,examples:[{input:"V=[0,0,0], s=0, r=1, s'=1, α=0.1, γ=0.9, done=False",output:"V=[0.1, 0, 0]",explanation:"V[0] += 0.1 * (1 + 0.9*0 - 0) = 0.1"}],starterCode:`import numpy as np

def td0_update(V: np.ndarray, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> np.ndarray:
    """
    Perform one TD(0) value update.

    Args:
        V: Value function array V(s) for each state
        state: Current state s
        reward: Reward received r
        next_state: Next state s'
        alpha: Learning rate
        gamma: Discount factor
        done: Whether next_state is terminal

    Returns:
        Updated value function
    """
    V = np.array(V, dtype=float)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Non-terminal transition",input:"([0, 0, 0], 0, 1.0, 1, 0.1, 0.9, False)",expected:"[0.1, 0.0, 0.0]",hidden:!1},{id:"2",description:"Terminal state",input:"([0, 0, 0], 0, 10.0, 1, 0.5, 0.9, True)",expected:"[5.0, 0.0, 0.0]",hidden:!1},{id:"3",description:"With existing values",input:"([5, 10, 15], 1, 2.0, 2, 0.1, 0.9, False)",expected:"[5.0, 10.55, 15.0]",hidden:!0}],hints:["If done=True, the next state value is 0 (terminal)","TD target: reward + gamma * V[next_state] (or just reward if done)","TD error: target - V[state]","Update: V[state] += alpha * td_error"],solution:`import numpy as np

def td0_update(V: np.ndarray, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> np.ndarray:
    """
    Perform one TD(0) value update.
    """
    V = np.array(V, dtype=float)

    # Next state value (0 if terminal)
    next_value = 0.0 if done else V[next_state]

    # TD target
    td_target = reward + gamma * next_value

    # TD error
    td_error = td_target - V[state]

    # Update value
    V[state] += alpha * td_error

    return np.round(V, 2).tolist()
`},{id:"rl-value-iteration",title:"Value Iteration Step",section:"reinforcement-learning",difficulty:"medium",description:`
## Value Iteration

Implement one step of the value iteration algorithm for solving MDPs.

### Bellman Optimality Update
\`\`\`
V(s) = max_a Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
\`\`\`

For each state, we find the action that maximizes expected value.

### Algorithm
1. For each state s:
   - For each action a, compute Q(s,a)
   - V(s) = max over all Q(s,a)
2. Repeat until convergence

### Function Signature
\`\`\`python
def value_iteration_step(V: np.ndarray, transitions: dict,
                         gamma: float) -> np.ndarray:
\`\`\`

Where \`transitions[s][a]\` is a list of \`(prob, reward, next_state)\` tuples.
    `,examples:[{input:"V=[0,0], transitions for 2-state MDP, gamma=0.9",output:"Updated V with optimal values",explanation:"Each state updated to max expected value over actions"}],starterCode:`import numpy as np

def value_iteration_step(V: np.ndarray, transitions: dict,
                         gamma: float) -> np.ndarray:
    """
    Perform one step of value iteration.

    Args:
        V: Current value function V(s) for each state
        transitions: Dict where transitions[s][a] = [(prob, reward, next_state), ...]
        gamma: Discount factor

    Returns:
        Updated value function
    """
    V = np.array(V, dtype=float)
    n_states = len(V)
    V_new = np.zeros(n_states)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple 2-state MDP",input:"([0.0, 0.0], {0: {0: [(1.0, 1.0, 0)], 1: [(1.0, 5.0, 1)]}, 1: {0: [(1.0, 2.0, 0)], 1: [(1.0, 0.0, 1)]}}, 0.9)",expected:"[5.0, 2.0]",hidden:!1},{id:"2",description:"With existing values",input:"([5.0, 2.0], {0: {0: [(1.0, 1.0, 0)], 1: [(1.0, 5.0, 1)]}, 1: {0: [(1.0, 2.0, 0)], 1: [(1.0, 0.0, 1)]}}, 0.9)",expected:"[6.8, 6.5]",hidden:!1}],hints:["For each state, compute the value of each action","Action value: sum over transitions of prob * (reward + gamma * V[next])","State value: max over all action values","Use the new values, not the old ones being updated"],solution:`import numpy as np

def value_iteration_step(V: np.ndarray, transitions: dict,
                         gamma: float) -> np.ndarray:
    """
    Perform one step of value iteration.
    """
    V = np.array(V, dtype=float)
    n_states = len(V)
    V_new = np.zeros(n_states)

    for s in range(n_states):
        if s not in transitions:
            continue

        # Compute Q-value for each action
        q_values = []
        for a in transitions[s]:
            q_sa = 0.0
            for prob, reward, next_state in transitions[s][a]:
                q_sa += prob * (reward + gamma * V[next_state])
            q_values.append(q_sa)

        # V(s) = max Q(s, a)
        V_new[s] = max(q_values) if q_values else 0.0

    return np.round(V_new, 1).tolist()
`},{id:"rl-policy-gradient",title:"REINFORCE Policy Gradient",section:"reinforcement-learning",difficulty:"hard",description:`
## REINFORCE (Monte Carlo Policy Gradient)

Implement the REINFORCE algorithm for computing policy gradient updates.

### Policy Gradient Theorem
\`\`\`
∇J(θ) = E[Σ_t ∇log π(a_t|s_t; θ) * G_t]
\`\`\`

### REINFORCE Algorithm
1. Collect episode: (s_0, a_0, r_1, s_1, a_1, r_2, ...)
2. For each timestep t:
   - Compute return G_t (discounted sum of future rewards)
   - Compute gradient: ∇log π(a_t|s_t) * G_t
3. Update policy parameters

### Softmax Policy
\`\`\`
π(a|s) = exp(θ[s,a]) / Σ_a' exp(θ[s,a'])
\`\`\`

### Gradient of Log-Softmax
\`\`\`
∇_θ log π(a|s) = one_hot(a) - π(·|s)
\`\`\`
(1 for action taken, minus probability of each action)

### Function Signature
\`\`\`python
def reinforce_gradient(theta: np.ndarray, episode: list,
                       gamma: float) -> np.ndarray:
\`\`\`

Episode is list of (state, action, reward) tuples.

### Expected Return Format
Returns gradient array of same shape as theta (n_states, n_actions).
    `,examples:[{input:"theta (3x2), episode=[(0,1,1), (1,0,2)], gamma=0.9",output:"Gradient array (3x2)",explanation:"Gradient computed from policy gradient theorem"}],starterCode:`import numpy as np

def softmax(x):
    """Compute softmax probabilities."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def reinforce_gradient(theta: np.ndarray, episode: list,
                       gamma: float) -> np.ndarray:
    """
    Compute REINFORCE policy gradient.

    Args:
        theta: Policy parameters (n_states, n_actions)
        episode: List of (state, action, reward) tuples
        gamma: Discount factor

    Returns:
        Gradient of same shape as theta
    """
    theta = np.array(theta, dtype=float)
    n_states, n_actions = theta.shape
    gradient = np.zeros_like(theta)

    # Your code here
    # 1. Compute returns G_t for each timestep
    # 2. For each (s, a, r), compute gradient contribution
    # 3. Gradient = (one_hot(a) - softmax(theta[s])) * G_t
    pass
`,testCases:[{id:"1",description:"Simple episode",input:"bool(np.allclose(reinforce_gradient(np.zeros((2, 2)), [(0, 1, 1.0), (1, 0, 1.0)], 0.9), [[-0.95, 0.95], [0.5, -0.5]], atol=0.01))",expected:"True",hidden:!1},{id:"2",description:"Single step episode",input:"bool(np.allclose(reinforce_gradient(np.zeros((2, 2)), [(0, 0, 5.0)], 0.9), [[2.5, -2.5], [0.0, 0.0]], atol=0.01))",expected:"True",hidden:!1}],hints:["First compute all returns: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...","For softmax policy: gradient = one_hot(action) - π(·|state)","Multiply gradient by return G_t and accumulate","Use backward iteration to compute returns efficiently"],solution:`import numpy as np

def softmax(x):
    """Compute softmax probabilities."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def reinforce_gradient(theta: np.ndarray, episode: list,
                       gamma: float) -> np.ndarray:
    """
    Compute REINFORCE policy gradient.
    """
    theta = np.array(theta, dtype=float)
    n_states, n_actions = theta.shape
    gradient = np.zeros_like(theta)

    T = len(episode)
    if T == 0:
        return gradient

    # Compute returns for each timestep (backward)
    returns = np.zeros(T)
    G = 0
    for t in range(T - 1, -1, -1):
        _, _, reward = episode[t]
        G = reward + gamma * G
        returns[t] = G

    # Compute gradient
    for t in range(T):
        state, action, _ = episode[t]
        G_t = returns[t]

        # Policy probabilities for this state
        probs = softmax(theta[state])

        # Gradient of log π(a|s) = one_hot(a) - π(·|s)
        grad_log_pi = -probs.copy()
        grad_log_pi[action] += 1.0

        # Accumulate: ∇log π(a|s) * G_t
        gradient[state] += grad_log_pi * G_t

    return np.round(gradient, 2)
`},{id:"rl-advantage-estimation",title:"Advantage Estimation (GAE)",section:"reinforcement-learning",difficulty:"hard",description:`
## Generalized Advantage Estimation (GAE)

Implement GAE, a crucial technique used in PPO and other policy gradient methods.

### Why Advantage?
The advantage function measures how much better an action is compared to average:
\`\`\`
A(s, a) = Q(s, a) - V(s)
\`\`\`

### GAE Formula
\`\`\`
Â_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}
\`\`\`

Where δ_t is the TD error:
\`\`\`
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
\`\`\`

### Lambda (λ) Trade-off
- λ = 0: Uses only one-step TD (low variance, high bias)
- λ = 1: Uses full Monte Carlo return (high variance, low bias)
- Typical: λ = 0.95

### Function Signature
\`\`\`python
def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
\`\`\`

Where values has length T+1 (includes V(s_T+1), use 0 if terminal).
    `,examples:[{input:"rewards=[1,1,1], values=[0,0,0,0], gamma=0.99, lambda=0.95",output:"advantages=[2.9, 1.95, 1.0]",explanation:"GAE combines TD errors with exponential decay"}],starterCode:`import numpy as np

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: Rewards r_0, r_1, ..., r_{T-1} (length T)
        values: Values V(s_0), V(s_1), ..., V(s_T) (length T+1)
                Last value is V(s_T) or 0 if terminal
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        Advantages Â_0, Â_1, ..., Â_{T-1} (length T)
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)
    advantages = np.zeros(T)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Zero values baseline",input:"bool(np.allclose(compute_gae([1, 1, 1], [0, 0, 0, 0], 0.99, 0.95), [2.9, 1.95, 1.0], atol=0.1))",expected:"True",hidden:!1},{id:"2",description:"With value estimates",input:"bool(np.allclose(compute_gae([1, 2, 3], [1, 2, 3, 0], 0.9, 0.9), [3.99, 2.7, 0.0], atol=0.1))",expected:"True",hidden:!1},{id:"3",description:"Single step",input:"bool(np.allclose(compute_gae([5], [1, 0], 0.99, 0.95), [4.0], atol=0.01))",expected:"True",hidden:!0}],hints:["Compute TD errors first: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)","Work backwards from the last timestep","GAE formula: Â_t = δ_t + γλ*Â_{t+1}","At the last timestep: Â_{T-1} = δ_{T-1}"],solution:`import numpy as np

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation.
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)
    advantages = np.zeros(T)

    # Compute advantages backwards
    gae = 0.0
    for t in range(T - 1, -1, -1):
        # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] - values[t]

        # GAE: Â_t = δ_t + γλ*Â_{t+1}
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return np.round(advantages, 2)
`},{id:"rl-ppo-clip",title:"PPO Clipped Objective",section:"reinforcement-learning",difficulty:"hard",description:`
## PPO Clipped Objective

Implement the clipped surrogate objective used in Proximal Policy Optimization (PPO).

### PPO Objective
\`\`\`
L^{CLIP}(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
\`\`\`

Where:
- r_t(θ) = π_θ(a|s) / π_θ_old(a|s) is the probability ratio
- A_t is the advantage estimate
- ε is the clipping parameter (typically 0.2)

### Why Clipping?
- Prevents too large policy updates
- Keeps new policy "close" to old policy
- More stable training than vanilla policy gradient

### Key Insight
- If A > 0 (good action): clip ratio to ≤ 1+ε
- If A < 0 (bad action): clip ratio to ≥ 1-ε

### Function Signature
\`\`\`python
def ppo_clip_objective(old_probs: np.ndarray, new_probs: np.ndarray,
                       advantages: np.ndarray, epsilon: float) -> float:
\`\`\`
    `,examples:[{input:"old_probs=[0.5], new_probs=[0.6], advantages=[1.0], epsilon=0.2",output:"1.0",explanation:"Ratio=1.2, clipped to 1.2, advantage positive: min(1.2, 1.2)*1=1.2, but capped at 1.0 example"}],starterCode:`import numpy as np

def ppo_clip_objective(old_probs: np.ndarray, new_probs: np.ndarray,
                       advantages: np.ndarray, epsilon: float) -> float:
    """
    Compute PPO clipped surrogate objective.

    Args:
        old_probs: π_old(a|s) probabilities for taken actions
        new_probs: π_new(a|s) probabilities for taken actions
        advantages: Advantage estimates A_t
        epsilon: Clipping parameter (e.g., 0.2)

    Returns:
        PPO clipped objective (scalar, should be maximized)
    """
    old_probs = np.array(old_probs, dtype=float)
    new_probs = np.array(new_probs, dtype=float)
    advantages = np.array(advantages, dtype=float)

    # Your code here
    pass
`,testCases:[{id:"1",description:"No clipping needed",input:"([0.5, 0.5], [0.5, 0.5], [1.0, -1.0], 0.2)",expected:"0.0",hidden:!1},{id:"2",description:"Positive advantage, ratio > 1",input:"([0.5], [0.7], [1.0], 0.2)",expected:"1.2",hidden:!1},{id:"3",description:"Negative advantage, ratio > 1",input:"([0.5], [0.7], [-1.0], 0.2)",expected:"-1.4",hidden:!1},{id:"4",description:"Clipping in action",input:"([0.3], [0.6], [2.0], 0.2)",expected:"2.4",hidden:!0}],hints:["Compute probability ratios: r = new_probs / old_probs","Compute clipped ratios: clip(r, 1-epsilon, 1+epsilon)","Unclipped term: r * advantages","Clipped term: clipped_r * advantages","Take element-wise minimum, then mean"],solution:`import numpy as np

def ppo_clip_objective(old_probs: np.ndarray, new_probs: np.ndarray,
                       advantages: np.ndarray, epsilon: float) -> float:
    """
    Compute PPO clipped surrogate objective.
    """
    old_probs = np.array(old_probs, dtype=float)
    new_probs = np.array(new_probs, dtype=float)
    advantages = np.array(advantages, dtype=float)

    # Probability ratios
    ratios = new_probs / (old_probs + 1e-8)

    # Clipped ratios
    clipped_ratios = np.clip(ratios, 1 - epsilon, 1 + epsilon)

    # Surrogate objectives
    surr1 = ratios * advantages
    surr2 = clipped_ratios * advantages

    # PPO objective: min of clipped and unclipped
    objective = np.mean(np.minimum(surr1, surr2))

    return round(objective, 2)
`},{id:"rl-n-step-return",title:"N-Step Return",section:"reinforcement-learning",difficulty:"medium",description:`
## N-Step Return

Implement n-step returns, which interpolate between TD(0) and Monte Carlo.

### N-Step Return Formula
\`\`\`
G_t^{(n)} = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * V(s_{t+n})
\`\`\`

### Trade-offs
| n | Method | Bias | Variance |
|---|--------|------|----------|
| 1 | TD(0) | High | Low |
| n | n-step | Medium | Medium |
| ∞ | MC | None | High |

### Function Signature
\`\`\`python
def n_step_return(rewards: np.ndarray, values: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
\`\`\`

Compute n-step return starting from timestep t.
    `,examples:[{input:"rewards=[1,2,3,4], values=[0,0,0,0,10], t=0, n=2, gamma=0.9",output:"2.8",explanation:"G_0^(2) = 1 + 0.9*2 + 0.81*0 = 2.8"}],starterCode:`import numpy as np

def n_step_return(rewards: np.ndarray, values: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
    """
    Compute n-step return from timestep t.

    Args:
        rewards: All rewards [r_0, r_1, ..., r_{T-1}]
        values: All values [V(s_0), ..., V(s_T)]
        t: Starting timestep
        n: Number of steps
        gamma: Discount factor

    Returns:
        N-step return G_t^(n)
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)

    # Your code here
    pass
`,testCases:[{id:"1",description:"2-step return",input:"([1, 2, 3, 4], [0, 0, 0, 0, 10], 0, 2, 0.9)",expected:"2.8",hidden:!1},{id:"2",description:"3-step return with bootstrap",input:"([1, 1, 1, 1], [0, 0, 0, 5, 0], 0, 3, 0.9)",expected:"6.36",hidden:!1},{id:"3",description:"1-step (TD) return",input:"([5, 2, 3], [1, 10, 5, 0], 0, 1, 0.9)",expected:"14.0",hidden:!0}],hints:["Sum discounted rewards from t to min(t+n-1, T-1)","Bootstrap with V(s_{t+n}) if t+n <= T","Handle boundary: if t+n > T, just sum remaining rewards","Discounts: γ^0, γ^1, ..., γ^{n-1} for rewards, γ^n for value"],solution:`import numpy as np

def n_step_return(rewards: np.ndarray, values: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
    """
    Compute n-step return from timestep t.
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)

    G = 0.0

    # Sum discounted rewards
    for k in range(n):
        if t + k < T:
            G += (gamma ** k) * rewards[t + k]

    # Bootstrap with value if we haven't reached terminal
    if t + n < len(values):
        G += (gamma ** n) * values[t + n]

    return round(G, 2)
`},{id:"rl-reward-modeling",title:"RLHF Reward Model Loss",section:"reinforcement-learning",difficulty:"hard",description:`
## RLHF Reward Model Loss

Implement the Bradley-Terry preference model loss used to train reward models in RLHF.

### Background
In RLHF (Reinforcement Learning from Human Feedback), we first train a reward model
on human preference data, then use it to fine-tune an LLM.

### Bradley-Terry Model
Given a pair of responses (y_w, y_l) where y_w is preferred (winner):
\`\`\`
P(y_w > y_l) = σ(r(y_w) - r(y_l))
\`\`\`

Where σ is the sigmoid function and r(y) is the reward model output.

### Loss Function
\`\`\`
L = -E[log σ(r(y_w) - r(y_l))]
\`\`\`

We want to maximize the probability that the reward model correctly ranks preferences.

### Function Signature
\`\`\`python
def reward_model_loss(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
\`\`\`

Where r_chosen and r_rejected are reward scores for preferred and rejected responses.
    `,examples:[{input:"r_chosen = [2.0, 1.5], r_rejected = [1.0, 0.5]",output:"0.313",explanation:"Low loss when chosen has higher reward than rejected"},{input:"r_chosen = [0.0], r_rejected = [0.0]",output:"0.693",explanation:"Equal rewards = 50% probability = -log(0.5) = 0.693"}],starterCode:`import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reward_model_loss(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
    """
    Compute RLHF reward model loss (Bradley-Terry).

    Args:
        r_chosen: Reward scores for preferred/chosen responses
        r_rejected: Reward scores for rejected responses

    Returns:
        Average negative log-likelihood loss
    """
    r_chosen = np.array(r_chosen, dtype=float)
    r_rejected = np.array(r_rejected, dtype=float)

    # Your code here
    # Loss = -E[log σ(r_chosen - r_rejected)]
    pass
`,testCases:[{id:"1",description:"Clear preference",input:"([2.0, 1.5], [1.0, 0.5])",expected:"0.313",hidden:!1},{id:"2",description:"Equal scores",input:"([0.0], [0.0])",expected:"0.693",hidden:!1},{id:"3",description:"Large margin",input:"([5.0, 4.0, 3.0], [0.0, 0.0, 0.0])",expected:"0.024",hidden:!0}],hints:["Compute the difference: r_chosen - r_rejected","Apply sigmoid to get probability","Take log of probability","Return negative mean (loss to minimize)"],solution:`import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reward_model_loss(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
    """
    Compute RLHF reward model loss (Bradley-Terry).
    """
    r_chosen = np.array(r_chosen, dtype=float)
    r_rejected = np.array(r_rejected, dtype=float)

    # Probability of correctly ranking: σ(r_chosen - r_rejected)
    logits = r_chosen - r_rejected
    probs = sigmoid(logits)

    # Negative log-likelihood
    loss = -np.mean(np.log(probs + 1e-10))

    return round(loss, 3)
`},{id:"rl-dpo-loss",title:"Direct Preference Optimization (DPO)",section:"reinforcement-learning",difficulty:"hard",description:`
## Direct Preference Optimization (DPO) Loss

Implement the DPO loss function, a simpler alternative to RLHF that doesn't require a separate reward model.

### Background
DPO directly optimizes the LLM policy using preference data, bypassing the reward modeling step.

### DPO Loss
\`\`\`
L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                    - log π_θ(y_l|x) + log π_ref(y_l|x)))]
\`\`\`

Simplified:
\`\`\`
L_DPO = -E[log σ(β * (Δ_w - Δ_l))]
\`\`\`
Where Δ = log π_θ(y|x) - log π_ref(y|x) (log-ratio of policy to reference)

### Parameters
- β: Temperature parameter (typically 0.1 to 0.5)
- π_θ: Policy being trained
- π_ref: Reference policy (frozen)
- y_w: Preferred (winner) response
- y_l: Rejected (loser) response

### Function Signature
\`\`\`python
def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray,
             ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray,
             beta: float) -> float:
\`\`\`
    `,examples:[{input:"logp_chosen=[-1], logp_rejected=[-2], ref_logp_chosen=[-1.5], ref_logp_rejected=[-1.5], beta=0.1",output:"0.643",explanation:"Policy prefers chosen over reference"}],starterCode:`import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray,
             ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray,
             beta: float) -> float:
    """
    Compute Direct Preference Optimization (DPO) loss.

    Args:
        logp_chosen: Log prob of chosen response under policy π_θ
        logp_rejected: Log prob of rejected response under policy π_θ
        ref_logp_chosen: Log prob of chosen under reference π_ref
        ref_logp_rejected: Log prob of rejected under reference π_ref
        beta: Temperature parameter

    Returns:
        DPO loss
    """
    logp_chosen = np.array(logp_chosen, dtype=float)
    logp_rejected = np.array(logp_rejected, dtype=float)
    ref_logp_chosen = np.array(ref_logp_chosen, dtype=float)
    ref_logp_rejected = np.array(ref_logp_rejected, dtype=float)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Policy prefers chosen",input:"([-1.0], [-2.0], [-1.5], [-1.5], 0.1)",expected:"0.644",hidden:!1},{id:"2",description:"Equal preferences",input:"([-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], 0.5)",expected:"0.693",hidden:!1},{id:"3",description:"Strong preference signal",input:"([0.0], [-5.0], [-2.0], [-3.0], 0.1)",expected:"0.513",hidden:!0}],hints:["Compute log-ratios: Δ_w = logp_chosen - ref_logp_chosen","Compute log-ratios: Δ_l = logp_rejected - ref_logp_rejected","DPO logits: β * (Δ_w - Δ_l)","Loss: -mean(log(sigmoid(logits)))"],solution:`import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray,
             ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray,
             beta: float) -> float:
    """
    Compute Direct Preference Optimization (DPO) loss.
    """
    logp_chosen = np.array(logp_chosen, dtype=float)
    logp_rejected = np.array(logp_rejected, dtype=float)
    ref_logp_chosen = np.array(ref_logp_chosen, dtype=float)
    ref_logp_rejected = np.array(ref_logp_rejected, dtype=float)

    # Log-ratios (policy vs reference)
    chosen_logratios = logp_chosen - ref_logp_chosen
    rejected_logratios = logp_rejected - ref_logp_rejected

    # DPO logits
    logits = beta * (chosen_logratios - rejected_logratios)

    # Negative log-sigmoid loss
    loss = -np.mean(np.log(sigmoid(logits) + 1e-10))

    return round(loss, 3)
`},{id:"rl-kl-penalty",title:"KL Divergence Penalty (RLHF)",section:"reinforcement-learning",difficulty:"medium",description:`
## KL Divergence Penalty for RLHF

Compute the KL divergence penalty used in RLHF to keep the policy close to the reference model.

### Background
During RLHF fine-tuning, we add a KL penalty to prevent the policy from diverging
too far from the original pretrained model:

\`\`\`
reward_total = reward - β * KL(π_θ || π_ref)
\`\`\`

### KL Divergence (per token)
\`\`\`
KL = E[log π_θ(a|s) - log π_ref(a|s)]
   = E[log(π_θ/π_ref)]
\`\`\`

For a sequence, we typically compute the average KL per token.

### Why KL Penalty?
- Prevents reward hacking
- Maintains model coherence
- Avoids mode collapse
- Keeps outputs in distribution

### Function Signature
\`\`\`python
def kl_penalty(logp_policy: np.ndarray, logp_reference: np.ndarray,
               beta: float) -> float:
\`\`\`

Returns: KL divergence multiplied by β
    `,examples:[{input:"logp_policy = [-1, -2], logp_reference = [-1.5, -2.5], beta = 0.1",output:"0.05",explanation:"KL = mean([0.5, 0.5]) = 0.5, penalty = 0.1 * 0.5 = 0.05"}],starterCode:`import numpy as np

def kl_penalty(logp_policy: np.ndarray, logp_reference: np.ndarray,
               beta: float) -> float:
    """
    Compute KL divergence penalty for RLHF.

    Args:
        logp_policy: Log probs from current policy π_θ
        logp_reference: Log probs from reference policy π_ref
        beta: KL penalty coefficient

    Returns:
        KL penalty = β * mean(KL divergence)
    """
    logp_policy = np.array(logp_policy, dtype=float)
    logp_reference = np.array(logp_reference, dtype=float)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Positive divergence",input:"([-1.0, -2.0], [-1.5, -2.5], 0.1)",expected:"0.05",hidden:!1},{id:"2",description:"No divergence",input:"([-1.0, -1.0], [-1.0, -1.0], 0.1)",expected:"0.0",hidden:!1},{id:"3",description:"Higher beta",input:"([-0.5, -1.0, -1.5], [-1.0, -1.5, -2.0], 0.5)",expected:"0.25",hidden:!0}],hints:["KL divergence per token: log(π_θ) - log(π_ref)","Take the mean over all tokens","Multiply by β to get the penalty","KL should be non-negative (though approximation may give small negatives)"],solution:`import numpy as np

def kl_penalty(logp_policy: np.ndarray, logp_reference: np.ndarray,
               beta: float) -> float:
    """
    Compute KL divergence penalty for RLHF.
    """
    logp_policy = np.array(logp_policy, dtype=float)
    logp_reference = np.array(logp_reference, dtype=float)

    # KL divergence: log(π_θ/π_ref) = log(π_θ) - log(π_ref)
    kl_div = logp_policy - logp_reference

    # Mean KL (per token)
    mean_kl = np.mean(kl_div)

    # KL penalty
    penalty = beta * mean_kl

    return round(penalty, 2)
`},{id:"rl-ppo-llm-objective",title:"PPO-RLHF Objective",section:"reinforcement-learning",difficulty:"hard",description:`
## PPO Objective for LLM Fine-tuning (RLHF)

Implement the full PPO-RLHF objective that combines the clipped surrogate, reward, and KL penalty.

### RLHF PPO Objective
\`\`\`
L = E[min(r_t * Â_t, clip(r_t, 1-ε, 1+ε) * Â_t) - β * KL]
\`\`\`

Where:
- r_t = π_θ(a|s) / π_old(a|s) is the probability ratio
- Â_t is the advantage (reward - baseline, typically from reward model)
- ε is the PPO clipping parameter (0.2)
- β is the KL penalty coefficient
- KL = log π_θ - log π_ref

### RLHF Advantage
In RLHF, the advantage is often simplified to:
\`\`\`
Â = R(x, y) - β * KL(π_θ || π_ref)
\`\`\`

### Function Signature
\`\`\`python
def ppo_rlhf_objective(old_logp: np.ndarray, new_logp: np.ndarray,
                       ref_logp: np.ndarray, rewards: np.ndarray,
                       epsilon: float, beta: float) -> dict:
\`\`\`

### Expected Return Format
Return a dictionary with:
- \`'objective'\`: The PPO-RLHF objective value
- \`'policy_loss'\`: Clipped policy loss component
- \`'kl_penalty'\`: KL penalty component
    `,examples:[{input:"old_logp=[-2], new_logp=[-1.8], ref_logp=[-2], rewards=[1.0], ε=0.2, β=0.1",output:"{'objective': ..., 'policy_loss': ..., 'kl_penalty': ...}",explanation:"Combined objective with reward and KL penalty"}],starterCode:`import numpy as np

def ppo_rlhf_objective(old_logp: np.ndarray, new_logp: np.ndarray,
                       ref_logp: np.ndarray, rewards: np.ndarray,
                       epsilon: float, beta: float) -> dict:
    """
    Compute PPO-RLHF objective for LLM fine-tuning.

    Args:
        old_logp: Log probs from old policy (before update)
        new_logp: Log probs from current policy
        ref_logp: Log probs from reference (pretrained) model
        rewards: Rewards from reward model
        epsilon: PPO clipping parameter
        beta: KL penalty coefficient

    Returns:
        Dictionary with 'objective', 'policy_loss', 'kl_penalty'
    """
    old_logp = np.array(old_logp, dtype=float)
    new_logp = np.array(new_logp, dtype=float)
    ref_logp = np.array(ref_logp, dtype=float)
    rewards = np.array(rewards, dtype=float)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Positive reward, small update",input:"(lambda r: bool(abs(r['objective'] - 1.18) < 0.1 and abs(r['kl_penalty'] - 0.02) < 0.01))(ppo_rlhf_objective([-2.0], [-1.8], [-2.0], [1.0], 0.2, 0.1))",expected:"True",hidden:!1},{id:"2",description:"No policy change",input:"(lambda r: bool(abs(r['objective'] - 1.0) < 0.01 and abs(r['kl_penalty']) < 0.01))(ppo_rlhf_objective([-1.0], [-1.0], [-1.0], [1.0], 0.2, 0.1))",expected:"True",hidden:!1}],hints:["Compute ratio: exp(new_logp - old_logp)","Compute KL: new_logp - ref_logp","Compute advantage: rewards - beta * KL","Apply PPO clipping to ratio * advantage","Final objective: clipped_objective - beta * KL"],solution:`import numpy as np

def ppo_rlhf_objective(old_logp: np.ndarray, new_logp: np.ndarray,
                       ref_logp: np.ndarray, rewards: np.ndarray,
                       epsilon: float, beta: float) -> dict:
    """
    Compute PPO-RLHF objective for LLM fine-tuning.
    """
    old_logp = np.array(old_logp, dtype=float)
    new_logp = np.array(new_logp, dtype=float)
    ref_logp = np.array(ref_logp, dtype=float)
    rewards = np.array(rewards, dtype=float)

    # Probability ratio
    ratio = np.exp(new_logp - old_logp)

    # KL divergence from reference
    kl = new_logp - ref_logp

    # KL penalty
    kl_penalty = beta * np.mean(kl)

    # Advantage: reward minus KL penalty
    advantage = rewards - beta * kl

    # Clipped ratio
    clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)

    # PPO surrogate objectives
    surr1 = ratio * advantage
    surr2 = clipped_ratio * advantage

    # Policy loss (take minimum for conservative update)
    policy_loss = np.mean(np.minimum(surr1, surr2))

    # Final objective (maximize policy_loss, minimize kl_penalty)
    objective = policy_loss

    return {
        'objective': round(float(objective), 2),
        'policy_loss': round(float(policy_loss), 2),
        'kl_penalty': round(float(kl_penalty), 2),
    }
`}],Kg=[...kC,...SC,...CC,...EC,...zC,...TC,...PC,...NC,...jC,...LC,...AC,...RC,...IC,...OC,...DC];function FC(e){return Kg.find(t=>t.id===e)}function MC(e){return Kg.filter(t=>t.section===e)}function qC(){const{sectionId:e}=Th(),{getProblemProgress:t,getSectionProgress:n}=ra(),r=Mr.find(o=>o.id===e),i=e?MC(e):[];if(!r)return p.jsxs("div",{className:"text-center py-12",children:[p.jsx("h1",{className:"text-2xl font-bold text-gray-900 mb-4",children:"Section Not Found"}),p.jsx(ot,{to:"/",className:"text-primary-600 hover:text-primary-500",children:"Return to Home"})]});const a=n(r.id,r.problems.length);return p.jsxs("div",{className:"max-w-4xl mx-auto",children:[p.jsx(Hr,{title:r.title,description:`${r.description} Learn ${r.title.toLowerCase()} with ${r.problems.length} hands-on coding problems.`,canonical:`/section/${r.id}`}),p.jsxs("div",{className:"mb-8",children:[p.jsxs("div",{className:"flex items-center gap-4 mb-4",children:[p.jsx("div",{className:"w-14 h-14 bg-white border border-gray-200 rounded-xl flex items-center justify-center text-3xl shadow-sm",children:r.icon}),p.jsxs("div",{children:[p.jsx("h1",{className:"text-2xl font-bold text-gray-900",children:r.title}),p.jsx("p",{className:"text-gray-500",children:r.description})]})]}),p.jsxs("div",{className:"flex items-center gap-4",children:[p.jsx("div",{className:"flex-1 h-2 bg-gray-200 rounded-full overflow-hidden",children:p.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${a}%`}})}),p.jsxs("span",{className:"text-primary-600 font-medium",children:[a,"% complete"]})]})]}),p.jsxs("div",{className:"bg-white rounded-xl p-6 mb-8 border border-gray-200 shadow-sm",children:[p.jsx("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Introduction"}),p.jsx("div",{className:"prose prose-sm max-w-none",children:p.jsx(kg,{remarkPlugins:[Qg],components:{h1:({children:o})=>p.jsx("h1",{className:"text-xl font-bold text-gray-900 mt-0 mb-4",children:o}),h2:({children:o})=>p.jsx("h2",{className:"text-lg font-semibold text-gray-900 mt-6 mb-3",children:o}),h3:({children:o})=>p.jsx("h3",{className:"text-base font-medium text-gray-700 mt-4 mb-2",children:o}),p:({children:o})=>p.jsx("p",{className:"text-gray-600 mb-3 leading-relaxed",children:o}),pre:({children:o})=>p.jsx("pre",{className:"bg-gray-100 rounded-lg p-4 overflow-x-auto my-3 text-sm text-gray-800",children:o}),code:({className:o,children:s})=>!o&&!String(s).includes(`
`)?p.jsx("code",{className:"bg-gray-100 px-1.5 py-0.5 rounded text-primary-600 text-sm",children:s}):p.jsx("code",{className:"text-sm",children:s}),ul:({children:o})=>p.jsx("ul",{className:"list-disc list-inside text-gray-600 space-y-1 mb-3",children:o}),li:({children:o})=>p.jsx("li",{className:"text-gray-600",children:o}),strong:({children:o})=>p.jsx("strong",{className:"text-gray-900 font-semibold",children:o}),table:({children:o})=>p.jsx("div",{className:"overflow-x-auto my-4",children:p.jsx("table",{className:"min-w-full border-collapse border border-gray-300 text-sm",children:o})}),thead:({children:o})=>p.jsx("thead",{className:"bg-gray-100",children:o}),tbody:({children:o})=>p.jsx("tbody",{className:"divide-y divide-gray-200",children:o}),tr:({children:o})=>p.jsx("tr",{children:o}),th:({children:o})=>p.jsx("th",{className:"border border-gray-300 px-3 py-2 text-left font-semibold text-gray-700",children:o}),td:({children:o})=>p.jsx("td",{className:"border border-gray-300 px-3 py-2 text-gray-600",children:o})},children:r.introduction})})]}),p.jsxs("div",{children:[p.jsx("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Problems"}),p.jsx("div",{className:"space-y-3",children:i.map((o,s)=>{const l=t(r.id,o.id);return p.jsxs(ot,{to:`/problem/${r.id}/${o.id}`,className:"flex items-center gap-4 bg-white rounded-lg p-4 border border-gray-200 hover:border-primary-400 hover:shadow-sm transition-all group",children:[p.jsx("div",{className:`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${l.status==="completed"?"bg-green-100 text-green-600":l.status==="in_progress"?"bg-yellow-100 text-yellow-600":"bg-gray-100 text-gray-500"}`,children:l.status==="completed"?"✓":s+1}),p.jsxs("div",{className:"flex-1",children:[p.jsx("h3",{className:"text-gray-900 font-medium group-hover:text-primary-600 transition-colors",children:o.title}),p.jsxs("div",{className:"flex items-center gap-3 mt-1",children:[p.jsx("span",{className:`text-xs px-2 py-0.5 rounded ${o.difficulty==="easy"?"bg-green-100 text-green-700":o.difficulty==="medium"?"bg-yellow-100 text-yellow-700":"bg-red-100 text-red-700"}`,children:o.difficulty}),p.jsxs("span",{className:"text-gray-400 text-xs",children:[o.testCases.length," test cases"]})]})]}),p.jsx("div",{className:"text-gray-400 group-hover:text-primary-600 transition-colors",children:"→"})]},o.id)})})]})]})}var Yg={exports:{}},WC="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED",BC=WC,VC=BC;function Xg(){}function Gg(){}Gg.resetWarningCache=Xg;var UC=function(){function e(r,i,a,o,s,l){if(l!==VC){var u=new Error("Calling PropTypes validators directly is not supported by the `prop-types` package. Use PropTypes.checkPropTypes() to call them. Read more at http://fb.me/use-check-prop-types");throw u.name="Invariant Violation",u}}e.isRequired=e;function t(){return e}var n={array:e,bigint:e,bool:e,func:e,number:e,object:e,string:e,symbol:e,any:e,arrayOf:t,element:e,elementType:e,instanceOf:t,node:e,objectOf:t,oneOf:t,oneOfType:t,shape:t,exact:t,checkPropTypes:Gg,resetWarningCache:Xg};return n.PropTypes=n,n};Yg.exports=UC();var HC=Yg.exports;const oe=Jn(HC);var ft=typeof window<"u"?window:null,Pc=ft===null,Ki=Pc?void 0:ft.document,St="addEventListener",Ct="removeEventListener",Qs="getBoundingClientRect",li="_a",Et="_b",Qt="_c",Na="horizontal",zt=function(){return!1},$C=Pc?"calc":["","-webkit-","-moz-","-o-"].filter(function(e){var t=Ki.createElement("div");return t.style.cssText="width:"+e+"calc(9px)",!!t.style.length}).shift()+"calc",Zg=function(e){return typeof e=="string"||e instanceof String},Xp=function(e){if(Zg(e)){var t=Ki.querySelector(e);if(!t)throw new Error("Selector "+e+" did not match a DOM element");return t}return e},De=function(e,t,n){var r=e[t];return r!==void 0?r:n},ja=function(e,t,n,r){if(t){if(r==="end")return 0;if(r==="center")return e/2}else if(n){if(r==="start")return 0;if(r==="center")return e/2}return e},QC=function(e,t){var n=Ki.createElement("div");return n.className="gutter gutter-"+t,n},KC=function(e,t,n){var r={};return Zg(t)?r[e]=t:r[e]=$C+"("+t+"% - "+n+"px)",r},YC=function(e,t){var n;return n={},n[e]=t+"px",n},Gp=function(e,t){if(t===void 0&&(t={}),Pc)return{};var n=e,r,i,a,o,s,l;Array.from&&(n=Array.from(n));var u=Xp(n[0]),d=u.parentNode,c=getComputedStyle?getComputedStyle(d):null,f=c?c.flexDirection:null,m=De(t,"sizes")||n.map(function(){return 100/n.length}),y=De(t,"minSize",100),_=Array.isArray(y)?y:n.map(function(){return y}),k=De(t,"maxSize",1/0),h=Array.isArray(k)?k:n.map(function(){return k}),g=De(t,"expandToMin",!1),x=De(t,"gutterSize",10),C=De(t,"gutterAlign","center"),z=De(t,"snapOffset",30),S=Array.isArray(z)?z:n.map(function(){return z}),T=De(t,"dragInterval",1),N=De(t,"direction",Na),M=De(t,"cursor",N===Na?"col-resize":"row-resize"),w=De(t,"gutter",QC),I=De(t,"elementStyle",KC),W=De(t,"gutterStyle",YC);N===Na?(r="width",i="clientX",a="left",o="right",s="clientWidth"):N==="vertical"&&(r="height",i="clientY",a="top",o="bottom",s="clientHeight");function Y(q,L,D,H){var Ee=I(r,L,D,H);Object.keys(Ee).forEach(function(_e){q.style[_e]=Ee[_e]})}function ee(q,L,D){var H=W(r,L,D);Object.keys(H).forEach(function(Ee){q.style[Ee]=H[Ee]})}function j(){return l.map(function(q){return q.size})}function $(q){return"touches"in q?q.touches[0][i]:q[i]}function ne(q){var L=l[this.a],D=l[this.b],H=L.size+D.size;L.size=q/this.size*H,D.size=H-q/this.size*H,Y(L.element,L.size,this[Et],L.i),Y(D.element,D.size,this[Qt],D.i)}function O(q){var L,D=l[this.a],H=l[this.b];this.dragging&&(L=$(q)-this.start+(this[Et]-this.dragOffset),T>1&&(L=Math.round(L/T)*T),L<=D.minSize+D.snapOffset+this[Et]?L=D.minSize+this[Et]:L>=this.size-(H.minSize+H.snapOffset+this[Qt])&&(L=this.size-(H.minSize+this[Qt])),L>=D.maxSize-D.snapOffset+this[Et]?L=D.maxSize+this[Et]:L<=this.size-(H.maxSize-H.snapOffset+this[Qt])&&(L=this.size-(H.maxSize+this[Qt])),ne.call(this,L),De(t,"onDrag",zt)(j()))}function U(){var q=l[this.a].element,L=l[this.b].element,D=q[Qs](),H=L[Qs]();this.size=D[r]+H[r]+this[Et]+this[Qt],this.start=D[a],this.end=D[o]}function v(q){if(!getComputedStyle)return null;var L=getComputedStyle(q);if(!L)return null;var D=q[s];return D===0?null:(N===Na?D-=parseFloat(L.paddingLeft)+parseFloat(L.paddingRight):D-=parseFloat(L.paddingTop)+parseFloat(L.paddingBottom),D)}function B(q){var L=v(d);if(L===null||_.reduce(function(_e,Be){return _e+Be},0)>L)return q;var D=0,H=[],Ee=q.map(function(_e,Be){var Ot=L*_e/100,An=ja(x,Be===0,Be===q.length-1,C),Rn=_[Be]+An;return Ot<Rn?(D+=Rn-Ot,H.push(0),Rn):(H.push(Ot-Rn),Ot)});return D===0?q:Ee.map(function(_e,Be){var Ot=_e;if(D>0&&H[Be]-D>0){var An=Math.min(D,H[Be]-D);D-=An,Ot=_e-An}return Ot/L*100})}function Q(){var q=this,L=l[q.a].element,D=l[q.b].element;q.dragging&&De(t,"onDragEnd",zt)(j()),q.dragging=!1,ft[Ct]("mouseup",q.stop),ft[Ct]("touchend",q.stop),ft[Ct]("touchcancel",q.stop),ft[Ct]("mousemove",q.move),ft[Ct]("touchmove",q.move),q.stop=null,q.move=null,L[Ct]("selectstart",zt),L[Ct]("dragstart",zt),D[Ct]("selectstart",zt),D[Ct]("dragstart",zt),L.style.userSelect="",L.style.webkitUserSelect="",L.style.MozUserSelect="",L.style.pointerEvents="",D.style.userSelect="",D.style.webkitUserSelect="",D.style.MozUserSelect="",D.style.pointerEvents="",q.gutter.style.cursor="",q.parent.style.cursor="",Ki.body.style.cursor=""}function b(q){if(!("button"in q&&q.button!==0)){var L=this,D=l[L.a].element,H=l[L.b].element;L.dragging||De(t,"onDragStart",zt)(j()),q.preventDefault(),L.dragging=!0,L.move=O.bind(L),L.stop=Q.bind(L),ft[St]("mouseup",L.stop),ft[St]("touchend",L.stop),ft[St]("touchcancel",L.stop),ft[St]("mousemove",L.move),ft[St]("touchmove",L.move),D[St]("selectstart",zt),D[St]("dragstart",zt),H[St]("selectstart",zt),H[St]("dragstart",zt),D.style.userSelect="none",D.style.webkitUserSelect="none",D.style.MozUserSelect="none",D.style.pointerEvents="none",H.style.userSelect="none",H.style.webkitUserSelect="none",H.style.MozUserSelect="none",H.style.pointerEvents="none",L.gutter.style.cursor=M,L.parent.style.cursor=M,Ki.body.style.cursor=M,U.call(L),L.dragOffset=$(q)-L.end}}m=B(m);var ie=[];l=n.map(function(q,L){var D={element:Xp(q),size:m[L],minSize:_[L],maxSize:h[L],snapOffset:S[L],i:L},H;if(L>0&&(H={a:L-1,b:L,dragging:!1,direction:N,parent:d},H[Et]=ja(x,L-1===0,!1,C),H[Qt]=ja(x,!1,L===n.length-1,C),f==="row-reverse"||f==="column-reverse")){var Ee=H.a;H.a=H.b,H.b=Ee}if(L>0){var _e=w(L,N,D.element);ee(_e,x,L),H[li]=b.bind(H),_e[St]("mousedown",H[li]),_e[St]("touchstart",H[li]),d.insertBefore(_e,D.element),H.gutter=_e}return Y(D.element,D.size,ja(x,L===0,L===n.length-1,C),L),L>0&&ie.push(H),D});function we(q){var L=q.i===ie.length,D=L?ie[q.i-1]:ie[q.i];U.call(D);var H=L?D.size-q.minSize-D[Qt]:q.minSize+D[Et];ne.call(D,H)}l.forEach(function(q){var L=q.element[Qs]()[r];L<q.minSize&&(g?we(q):q.minSize=L)});function le(q){var L=B(q);L.forEach(function(D,H){if(H>0){var Ee=ie[H-1],_e=l[Ee.a],Be=l[Ee.b];_e.size=L[H-1],Be.size=D,Y(_e.element,_e.size,Ee[Et],_e.i),Y(Be.element,Be.size,Ee[Qt],Be.i)}})}function pt(q,L){ie.forEach(function(D){if(L!==!0?D.parent.removeChild(D.gutter):(D.gutter[Ct]("mousedown",D[li]),D.gutter[Ct]("touchstart",D[li])),q!==!0){var H=I(r,D.a.size,D[Et]);Object.keys(H).forEach(function(Ee){l[D.a].element.style[Ee]="",l[D.b].element.style[Ee]=""})}})}return{setSizes:le,getSizes:j,collapse:function(L){we(l[L])},destroy:pt,parent:d,pairs:ie}};function Ks(e,t){var n={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&t.indexOf(r)===-1&&(n[r]=e[r]);return n}var Nr=function(e){function t(){e.apply(this,arguments)}return e&&(t.__proto__=e),t.prototype=Object.create(e&&e.prototype),t.prototype.constructor=t,t.prototype.componentDidMount=function(){var r=this.props;r.children;var i=r.gutter,a=Ks(r,["children","gutter"]),o=a;o.gutter=function(s,l){var u;return i?u=i(s,l):(u=document.createElement("div"),u.className="gutter gutter-"+l),u.__isSplitGutter=!0,u},this.split=Gp(this.parent.children,o)},t.prototype.componentDidUpdate=function(r){var i=this,a=this.props;a.children;var o=a.minSize,s=a.sizes,l=a.collapsed,u=Ks(a,["children","minSize","sizes","collapsed"]),d=u,c=r.minSize,f=r.sizes,m=r.collapsed,y=["maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor"],_=y.map(function(g){return i.props[g]!==r[g]}).reduce(function(g,x){return g||x},!1);if(Array.isArray(o)&&Array.isArray(c)){var k=!1;o.forEach(function(g,x){k=k||g!==c[x]}),_=_||k}else Array.isArray(o)||Array.isArray(c)?_=!0:_=_||o!==c;if(_)d.minSize=o,d.sizes=s||this.split.getSizes(),this.split.destroy(!0,!0),d.gutter=function(g,x,C){return C.previousSibling},this.split=Gp(Array.from(this.parent.children).filter(function(g){return!g.__isSplitGutter}),d);else if(s){var h=!1;s.forEach(function(g,x){h=h||g!==f[x]}),h&&this.split.setSizes(this.props.sizes)}Number.isInteger(l)&&(l!==m||_)&&this.split.collapse(l)},t.prototype.componentWillUnmount=function(){this.split.destroy(),delete this.split},t.prototype.render=function(){var r=this,i=this.props;i.sizes,i.minSize,i.maxSize,i.expandToMin,i.gutterSize,i.gutterAlign,i.snapOffset,i.dragInterval,i.direction,i.cursor,i.gutter,i.elementStyle,i.gutterStyle,i.onDrag,i.onDragStart,i.onDragEnd,i.collapsed;var a=i.children,o=Ks(i,["sizes","minSize","maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor","gutter","elementStyle","gutterStyle","onDrag","onDragStart","onDragEnd","collapsed","children"]),s=o;return Ne.createElement("div",Object.assign({},{ref:function(l){r.parent=l}},s),a)},t}(Ne.Component);Nr.propTypes={sizes:oe.arrayOf(oe.number),minSize:oe.oneOfType([oe.number,oe.arrayOf(oe.number)]),maxSize:oe.oneOfType([oe.number,oe.arrayOf(oe.number)]),expandToMin:oe.bool,gutterSize:oe.number,gutterAlign:oe.string,snapOffset:oe.oneOfType([oe.number,oe.arrayOf(oe.number)]),dragInterval:oe.number,direction:oe.string,cursor:oe.string,gutter:oe.func,elementStyle:oe.func,gutterStyle:oe.func,onDrag:oe.func,onDragStart:oe.func,onDragEnd:oe.func,collapsed:oe.number,children:oe.arrayOf(oe.element)};Nr.defaultProps={sizes:void 0,minSize:void 0,maxSize:void 0,expandToMin:void 0,gutterSize:void 0,gutterAlign:void 0,snapOffset:void 0,dragInterval:void 0,direction:void 0,cursor:void 0,gutter:void 0,elementStyle:void 0,gutterStyle:void 0,onDrag:void 0,onDragStart:void 0,onDragEnd:void 0,collapsed:void 0,children:void 0};function Jg(){const[e,t]=E.useState(!1),[n,r]=E.useState(!1),[i,a]=E.useState(null),[o,s]=E.useState([]),l=E.useRef(null),u=E.useRef(!1),d=E.useCallback(_=>{s(k=>[...k,_])},[]),c=E.useCallback(()=>{s([])},[]),f=E.useCallback(async()=>{if(!(l.current||u.current)){u.current=!0,t(!0),a(null);try{window.loadPyodide||await new Promise((k,h)=>{const g=document.createElement("script");g.src="https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js",g.async=!0,g.onload=()=>k(),g.onerror=()=>h(new Error("Failed to load Pyodide")),document.head.appendChild(g)});const _=await window.loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.26.4/full/"});_.setStdout({batched:k=>{k.split(`
`).forEach(g=>{g.trim()&&d(g)})}}),_.setStderr({batched:k=>{k.split(`
`).forEach(g=>{g.trim()&&d(`[Error] ${g}`)})}}),await _.loadPackage(["numpy"]),l.current=_,r(!0),d("Python environment ready with NumPy loaded.")}catch(_){const k=_ instanceof Error?_.message:"Failed to initialize Python";a(k),d(`[Error] ${k}`)}finally{t(!1),u.current=!1}}},[d]);E.useEffect(()=>{f()},[f]);const m=E.useCallback(async _=>{if(!l.current)return{success:!1,error:"Python environment not ready"};try{return c(),await l.current.runPythonAsync(_),{success:!0}}catch(k){const h=k instanceof Error?k.message:"Execution error";return d(`[Error] ${h}`),{success:!1,error:h}}},[d,c]),y=E.useCallback(async(_,k,h)=>{if(!l.current)return k.map(x=>({id:x.id,passed:!1,description:x.description,expected:x.expected,actual:"Python environment not ready",hidden:x.hidden}));const g=[];c();try{await l.current.runPythonAsync(_);for(const x of k)try{const C=x.input.includes(h)||x.input.includes(".")||/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(x.input.trim()),z=x.input.trim().startsWith("(lambda");let S;C&&!x.input.startsWith("[")&&(!x.input.startsWith("(")||z)?S=`
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
    result = ${h}(np.array(test_input))
elif isinstance(test_input, tuple):
    result = ${h}(*[np.array(x) if isinstance(x, list) else x for x in test_input])
else:
    result = ${h}(test_input)

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
`;const T=await l.current.runPythonAsync(S),N=String(T),M=x.expected.replace(/\s/g,""),w=N.replace(/\s/g,""),I=M===w;g.push({id:x.id,passed:I,description:x.description,expected:x.expected,actual:N,hidden:x.hidden}),d(I?`Test ${x.id}: PASSED`:`Test ${x.id}: FAILED - Expected ${x.expected}, got ${N}`)}catch(C){const z=C instanceof Error?C.message:"Test execution error";g.push({id:x.id,passed:!1,description:x.description,expected:x.expected,actual:z,hidden:x.hidden}),d(`Test ${x.id}: ERROR - ${z}`)}}catch(x){const C=x instanceof Error?x.message:"Code execution error";return d(`[Error] ${C}`),k.map(z=>({id:z.id,passed:!1,description:z.description,expected:z.expected,actual:C,hidden:z.hidden}))}return g},[d,c]);return{isLoading:e,isReady:n,error:i,output:o,runCode:m,runTests:y,clearOutput:c}}function Zp(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}function XC(e){if(Array.isArray(e))return e}function GC(e,t,n){return(t=iE(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function ZC(e,t){var n=e==null?null:typeof Symbol<"u"&&e[Symbol.iterator]||e["@@iterator"];if(n!=null){var r,i,a,o,s=[],l=!0,u=!1;try{if(a=(n=n.call(e)).next,t!==0)for(;!(l=(r=a.call(n)).done)&&(s.push(r.value),s.length!==t);l=!0);}catch(d){u=!0,i=d}finally{try{if(!l&&n.return!=null&&(o=n.return(),Object(o)!==o))return}finally{if(u)throw i}}return s}}function JC(){throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function Jp(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function ef(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?Jp(Object(n),!0).forEach(function(r){GC(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):Jp(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function eE(e,t){if(e==null)return{};var n,r,i=tE(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)===-1&&{}.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}function tE(e,t){if(e==null)return{};var n={};for(var r in e)if({}.hasOwnProperty.call(e,r)){if(t.indexOf(r)!==-1)continue;n[r]=e[r]}return n}function nE(e,t){return XC(e)||ZC(e,t)||aE(e,t)||JC()}function rE(e,t){if(typeof e!="object"||!e)return e;var n=e[Symbol.toPrimitive];if(n!==void 0){var r=n.call(e,t);if(typeof r!="object")return r;throw new TypeError("@@toPrimitive must return a primitive value.")}return(t==="string"?String:Number)(e)}function iE(e){var t=rE(e,"string");return typeof t=="symbol"?t:t+""}function aE(e,t){if(e){if(typeof e=="string")return Zp(e,t);var n={}.toString.call(e).slice(8,-1);return n==="Object"&&e.constructor&&(n=e.constructor.name),n==="Map"||n==="Set"?Array.from(e):n==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?Zp(e,t):void 0}}function oE(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function tf(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function nf(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?tf(Object(n),!0).forEach(function(r){oE(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):tf(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function sE(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(r){return t.reduceRight(function(i,a){return a(i)},r)}}function fi(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),a=0;a<r;a++)i[a]=arguments[a];return i.length>=e.length?e.apply(this,i):function(){for(var o=arguments.length,s=new Array(o),l=0;l<o;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function To(e){return{}.toString.call(e).includes("Object")}function lE(e){return!Object.keys(e).length}function Yi(e){return typeof e=="function"}function uE(e,t){return Object.prototype.hasOwnProperty.call(e,t)}function cE(e,t){return To(t)||Sn("changeType"),Object.keys(t).some(function(n){return!uE(e,n)})&&Sn("changeField"),t}function dE(e){Yi(e)||Sn("selectorType")}function pE(e){Yi(e)||To(e)||Sn("handlerType"),To(e)&&Object.values(e).some(function(t){return!Yi(t)})&&Sn("handlersType")}function fE(e){e||Sn("initialIsRequired"),To(e)||Sn("initialType"),lE(e)&&Sn("initialContent")}function mE(e,t){throw new Error(e[t]||e.default)}var hE={initialIsRequired:"initial state is required",initialType:"initial state should be an object",initialContent:"initial state shouldn't be an empty object",handlerType:"handler should be an object or a function",handlersType:"all handlers should be a functions",selectorType:"selector should be a function",changeType:"provided value of changes should be an object",changeField:'it seams you want to change a field in the state which is not specified in the "initial" state',default:"an unknown error accured in `state-local` package"},Sn=fi(mE)(hE),La={changes:cE,selector:dE,handler:pE,initial:fE};function gE(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};La.initial(e),La.handler(t);var n={current:e},r=fi(_E)(n,t),i=fi(xE)(n),a=fi(La.changes)(e),o=fi(yE)(n);function s(){var u=arguments.length>0&&arguments[0]!==void 0?arguments[0]:function(d){return d};return La.selector(u),u(n.current)}function l(u){sE(r,i,a,o)(u)}return[s,l]}function yE(e,t){return Yi(t)?t(e.current):t}function xE(e,t){return e.current=nf(nf({},e.current),t),t}function _E(e,t,n){return Yi(t)?t(e.current):Object.keys(n).forEach(function(r){var i;return(i=t[r])===null||i===void 0?void 0:i.call(t,e.current[r])}),n}var vE={create:gE},bE={paths:{vs:"https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs"}};function wE(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),a=0;a<r;a++)i[a]=arguments[a];return i.length>=e.length?e.apply(this,i):function(){for(var o=arguments.length,s=new Array(o),l=0;l<o;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function kE(e){return{}.toString.call(e).includes("Object")}function SE(e){return e||rf("configIsRequired"),kE(e)||rf("configType"),e.urls?(CE(),{paths:{vs:e.urls.monacoBase}}):e}function CE(){console.warn(ey.deprecation)}function EE(e,t){throw new Error(e[t]||e.default)}var ey={configIsRequired:"the configuration object is required",configType:"the configuration object should be an object",default:"an unknown error accured in `@monaco-editor/loader` package",deprecation:`Deprecation warning!
    You are using deprecated way of configuration.

    Instead of using
      monaco.config({ urls: { monacoBase: '...' } })
    use
      monaco.config({ paths: { vs: '...' } })

    For more please check the link https://github.com/suren-atoyan/monaco-loader#config
  `},rf=wE(EE)(ey),zE={config:SE},TE=function(){for(var t=arguments.length,n=new Array(t),r=0;r<t;r++)n[r]=arguments[r];return function(i){return n.reduceRight(function(a,o){return o(a)},i)}};function ty(e,t){return Object.keys(t).forEach(function(n){t[n]instanceof Object&&e[n]&&Object.assign(t[n],ty(e[n],t[n]))}),ef(ef({},e),t)}var PE={type:"cancelation",msg:"operation is manually canceled"};function Ys(e){var t=!1,n=new Promise(function(r,i){e.then(function(a){return t?i(PE):r(a)}),e.catch(i)});return n.cancel=function(){return t=!0},n}var NE=["monaco"],jE=vE.create({config:bE,isInitialized:!1,resolve:null,reject:null,monaco:null}),ny=nE(jE,2),sa=ny[0],es=ny[1];function LE(e){var t=zE.config(e),n=t.monaco,r=eE(t,NE);es(function(i){return{config:ty(i.config,r),monaco:n}})}function AE(){var e=sa(function(t){var n=t.monaco,r=t.isInitialized,i=t.resolve;return{monaco:n,isInitialized:r,resolve:i}});if(!e.isInitialized){if(es({isInitialized:!0}),e.monaco)return e.resolve(e.monaco),Ys(Xs);if(window.monaco&&window.monaco.editor)return ry(window.monaco),e.resolve(window.monaco),Ys(Xs);TE(RE,OE)(DE)}return Ys(Xs)}function RE(e){return document.body.appendChild(e)}function IE(e){var t=document.createElement("script");return e&&(t.src=e),t}function OE(e){var t=sa(function(r){var i=r.config,a=r.reject;return{config:i,reject:a}}),n=IE("".concat(t.config.paths.vs,"/loader.js"));return n.onload=function(){return e()},n.onerror=t.reject,n}function DE(){var e=sa(function(n){var r=n.config,i=n.resolve,a=n.reject;return{config:r,resolve:i,reject:a}}),t=window.require;t.config(e.config),t(["vs/editor/editor.main"],function(n){var r=n.m||n;ry(r),e.resolve(r)},function(n){e.reject(n)})}function ry(e){sa().monaco||es({monaco:e})}function FE(){return sa(function(e){var t=e.monaco;return t})}var Xs=new Promise(function(e,t){return es({resolve:e,reject:t})}),iy={config:LE,init:AE,__getMonacoInstance:FE},ME={wrapper:{display:"flex",position:"relative",textAlign:"initial"},fullWidth:{width:"100%"},hide:{display:"none"}},Gs=ME,qE={container:{display:"flex",height:"100%",width:"100%",justifyContent:"center",alignItems:"center"}},WE=qE;function BE({children:e}){return Ne.createElement("div",{style:WE.container},e)}var VE=BE,UE=VE;function HE({width:e,height:t,isEditorReady:n,loading:r,_ref:i,className:a,wrapperProps:o}){return Ne.createElement("section",{style:{...Gs.wrapper,width:e,height:t},...o},!n&&Ne.createElement(UE,null,r),Ne.createElement("div",{ref:i,style:{...Gs.fullWidth,...!n&&Gs.hide},className:a}))}var $E=HE,ay=E.memo($E);function QE(e){E.useEffect(e,[])}var oy=QE;function KE(e,t,n=!0){let r=E.useRef(!0);E.useEffect(r.current||!n?()=>{r.current=!1}:e,t)}var it=KE;function Ei(){}function vr(e,t,n,r){return YE(e,r)||XE(e,t,n,r)}function YE(e,t){return e.editor.getModel(sy(e,t))}function XE(e,t,n,r){return e.editor.createModel(t,n,r?sy(e,r):void 0)}function sy(e,t){return e.Uri.parse(t)}function GE({original:e,modified:t,language:n,originalLanguage:r,modifiedLanguage:i,originalModelPath:a,modifiedModelPath:o,keepCurrentOriginalModel:s=!1,keepCurrentModifiedModel:l=!1,theme:u="light",loading:d="Loading...",options:c={},height:f="100%",width:m="100%",className:y,wrapperProps:_={},beforeMount:k=Ei,onMount:h=Ei}){let[g,x]=E.useState(!1),[C,z]=E.useState(!0),S=E.useRef(null),T=E.useRef(null),N=E.useRef(null),M=E.useRef(h),w=E.useRef(k),I=E.useRef(!1);oy(()=>{let j=iy.init();return j.then($=>(T.current=$)&&z(!1)).catch($=>($==null?void 0:$.type)!=="cancelation"&&console.error("Monaco initialization: error:",$)),()=>S.current?ee():j.cancel()}),it(()=>{if(S.current&&T.current){let j=S.current.getOriginalEditor(),$=vr(T.current,e||"",r||n||"text",a||"");$!==j.getModel()&&j.setModel($)}},[a],g),it(()=>{if(S.current&&T.current){let j=S.current.getModifiedEditor(),$=vr(T.current,t||"",i||n||"text",o||"");$!==j.getModel()&&j.setModel($)}},[o],g),it(()=>{let j=S.current.getModifiedEditor();j.getOption(T.current.editor.EditorOption.readOnly)?j.setValue(t||""):t!==j.getValue()&&(j.executeEdits("",[{range:j.getModel().getFullModelRange(),text:t||"",forceMoveMarkers:!0}]),j.pushUndoStop())},[t],g),it(()=>{var j,$;($=(j=S.current)==null?void 0:j.getModel())==null||$.original.setValue(e||"")},[e],g),it(()=>{let{original:j,modified:$}=S.current.getModel();T.current.editor.setModelLanguage(j,r||n||"text"),T.current.editor.setModelLanguage($,i||n||"text")},[n,r,i],g),it(()=>{var j;(j=T.current)==null||j.editor.setTheme(u)},[u],g),it(()=>{var j;(j=S.current)==null||j.updateOptions(c)},[c],g);let W=E.useCallback(()=>{var ne;if(!T.current)return;w.current(T.current);let j=vr(T.current,e||"",r||n||"text",a||""),$=vr(T.current,t||"",i||n||"text",o||"");(ne=S.current)==null||ne.setModel({original:j,modified:$})},[n,t,i,e,r,a,o]),Y=E.useCallback(()=>{var j;!I.current&&N.current&&(S.current=T.current.editor.createDiffEditor(N.current,{automaticLayout:!0,...c}),W(),(j=T.current)==null||j.editor.setTheme(u),x(!0),I.current=!0)},[c,u,W]);E.useEffect(()=>{g&&M.current(S.current,T.current)},[g]),E.useEffect(()=>{!C&&!g&&Y()},[C,g,Y]);function ee(){var $,ne,O,U;let j=($=S.current)==null?void 0:$.getModel();s||((ne=j==null?void 0:j.original)==null||ne.dispose()),l||((O=j==null?void 0:j.modified)==null||O.dispose()),(U=S.current)==null||U.dispose()}return Ne.createElement(ay,{width:m,height:f,isEditorReady:g,loading:d,_ref:N,className:y,wrapperProps:_})}var ZE=GE;E.memo(ZE);function JE(e){let t=E.useRef();return E.useEffect(()=>{t.current=e},[e]),t.current}var ez=JE,Aa=new Map;function tz({defaultValue:e,defaultLanguage:t,defaultPath:n,value:r,language:i,path:a,theme:o="light",line:s,loading:l="Loading...",options:u={},overrideServices:d={},saveViewState:c=!0,keepCurrentModel:f=!1,width:m="100%",height:y="100%",className:_,wrapperProps:k={},beforeMount:h=Ei,onMount:g=Ei,onChange:x,onValidate:C=Ei}){let[z,S]=E.useState(!1),[T,N]=E.useState(!0),M=E.useRef(null),w=E.useRef(null),I=E.useRef(null),W=E.useRef(g),Y=E.useRef(h),ee=E.useRef(),j=E.useRef(r),$=ez(a),ne=E.useRef(!1),O=E.useRef(!1);oy(()=>{let B=iy.init();return B.then(Q=>(M.current=Q)&&N(!1)).catch(Q=>(Q==null?void 0:Q.type)!=="cancelation"&&console.error("Monaco initialization: error:",Q)),()=>w.current?v():B.cancel()}),it(()=>{var Q,b,ie,we;let B=vr(M.current,e||r||"",t||i||"",a||n||"");B!==((Q=w.current)==null?void 0:Q.getModel())&&(c&&Aa.set($,(b=w.current)==null?void 0:b.saveViewState()),(ie=w.current)==null||ie.setModel(B),c&&((we=w.current)==null||we.restoreViewState(Aa.get(a))))},[a],z),it(()=>{var B;(B=w.current)==null||B.updateOptions(u)},[u],z),it(()=>{!w.current||r===void 0||(w.current.getOption(M.current.editor.EditorOption.readOnly)?w.current.setValue(r):r!==w.current.getValue()&&(O.current=!0,w.current.executeEdits("",[{range:w.current.getModel().getFullModelRange(),text:r,forceMoveMarkers:!0}]),w.current.pushUndoStop(),O.current=!1))},[r],z),it(()=>{var Q,b;let B=(Q=w.current)==null?void 0:Q.getModel();B&&i&&((b=M.current)==null||b.editor.setModelLanguage(B,i))},[i],z),it(()=>{var B;s!==void 0&&((B=w.current)==null||B.revealLine(s))},[s],z),it(()=>{var B;(B=M.current)==null||B.editor.setTheme(o)},[o],z);let U=E.useCallback(()=>{var B;if(!(!I.current||!M.current)&&!ne.current){Y.current(M.current);let Q=a||n,b=vr(M.current,r||e||"",t||i||"",Q||"");w.current=(B=M.current)==null?void 0:B.editor.create(I.current,{model:b,automaticLayout:!0,...u},d),c&&w.current.restoreViewState(Aa.get(Q)),M.current.editor.setTheme(o),s!==void 0&&w.current.revealLine(s),S(!0),ne.current=!0}},[e,t,n,r,i,a,u,d,c,o,s]);E.useEffect(()=>{z&&W.current(w.current,M.current)},[z]),E.useEffect(()=>{!T&&!z&&U()},[T,z,U]),j.current=r,E.useEffect(()=>{var B,Q;z&&x&&((B=ee.current)==null||B.dispose(),ee.current=(Q=w.current)==null?void 0:Q.onDidChangeModelContent(b=>{O.current||x(w.current.getValue(),b)}))},[z,x]),E.useEffect(()=>{if(z){let B=M.current.editor.onDidChangeMarkers(Q=>{var ie;let b=(ie=w.current.getModel())==null?void 0:ie.uri;if(b&&Q.find(we=>we.path===b.path)){let we=M.current.editor.getModelMarkers({resource:b});C==null||C(we)}});return()=>{B==null||B.dispose()}}return()=>{}},[z,C]);function v(){var B,Q;(B=ee.current)==null||B.dispose(),f?c&&Aa.set(a,w.current.saveViewState()):(Q=w.current.getModel())==null||Q.dispose(),w.current.dispose()}return Ne.createElement(ay,{width:m,height:y,isEditorReady:z,loading:l,_ref:I,className:_,wrapperProps:k})}var nz=tz,rz=E.memo(nz),iz=rz;function pu({value:e,onChange:t,height:n="400px",onRun:r}){const i=E.useRef(null),a=E.useRef(r);E.useEffect(()=>{a.current=r},[r]);const o=E.useCallback(l=>{t(l||"")},[t]),s=E.useCallback((l,u)=>{i.current=l,l.addAction({id:"run-tests",label:"Run Tests",keybindings:[u.KeyMod.Shift|u.KeyCode.Enter],run:()=>{a.current&&a.current()}})},[]);return p.jsx("div",{className:"h-full border border-gray-200 rounded-lg overflow-hidden shadow-sm",children:p.jsx(iz,{height:n,defaultLanguage:"python",theme:"light",value:e,onChange:o,onMount:s,options:{minimap:{enabled:!1},fontSize:14,lineNumbers:"on",scrollBeyondLastLine:!1,automaticLayout:!0,tabSize:4,wordWrap:"on",padding:{top:16,bottom:16},scrollbar:{verticalScrollbarSize:8,horizontalScrollbarSize:8}}})})}function az(e){const t=e.filter(n=>!n.hidden);return t.length===0?"":t.map((n,r)=>`# Test ${r+1}: ${n.description}
input: ${n.input}
expected: ${n.expected}`).join(`

`)}function oz(e,t){const n=t.filter(a=>a.hidden);return e.trim()?[...e.split(/\n\n+/).filter(a=>a.trim()).map((a,o)=>{const s=a.split(`
`);let l=`Test ${o+1}`,u="",d="";for(const c of s)if(c.startsWith("# Test")){const f=c.match(/# Test \d+: (.+)/);f&&(l=f[1])}else c.startsWith("input:")?u=c.replace("input:","").trim():c.startsWith("expected:")&&(d=c.replace("expected:","").trim());return{id:String(o+1),description:l,input:u,expected:d,hidden:!1}}).filter(a=>a.input&&a.expected),...n]:n}function fu({output:e,isLoading:t=!1,testCases:n,onTestCasesChange:r,problemId:i,resetKey:a=0}){const o=E.useRef(null),[s,l]=E.useState("tests"),[u,d]=E.useState(""),c=E.useRef(void 0),f=E.useRef(0),m=(n==null?void 0:n.filter(h=>!h.hidden).length)??0,y=(n==null?void 0:n.filter(h=>h.hidden).length)??0;E.useEffect(()=>{if(n&&n.length>0){const h=i!==c.current,g=a!==f.current,x=n.filter(z=>!z.hidden),C=u===""&&x.length>0;(h||g||C)&&(d(az(n)),c.current=i,f.current=a,h&&l("tests"))}},[n,i,a,u]),E.useEffect(()=>{e.length>0&&l("output")},[e]),E.useEffect(()=>{var h;s==="output"&&((h=o.current)==null||h.scrollIntoView({behavior:"smooth"}))},[e,s]);const _=h=>{if(d(h),r&&n){const g=oz(h,n);r(g)}},k=n&&m>0;return p.jsxs("div",{className:"bg-white border border-gray-200 rounded-lg h-full flex flex-col shadow-sm",children:[p.jsxs("div",{className:"flex items-center justify-between px-2 py-1 border-b border-gray-200 bg-gray-50",children:[p.jsxs("div",{className:"flex",children:[k&&p.jsxs("button",{onClick:()=>l("tests"),className:`px-3 py-1.5 text-sm font-medium rounded-t transition-colors ${s==="tests"?"text-primary-600 bg-white border-t border-l border-r border-gray-200 -mb-px":"text-gray-500 hover:text-gray-700"}`,children:["Tests (",m,")"]}),p.jsx("button",{onClick:()=>l("output"),className:`px-3 py-1.5 text-sm font-medium rounded-t transition-colors ${s==="output"?"text-primary-600 bg-white border-t border-l border-r border-gray-200 -mb-px":"text-gray-500 hover:text-gray-700"}`,children:"Output"})]}),t&&p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-primary-500 rounded-full animate-pulse"}),p.jsx("span",{className:"text-xs text-gray-500",children:"Running..."})]})]}),s==="tests"&&k?p.jsxs("div",{className:"flex-1 flex flex-col overflow-hidden",children:[p.jsx("textarea",{value:u,onChange:h=>_(h.target.value),className:"flex-1 p-4 font-mono text-sm bg-gray-900 text-gray-300 resize-none focus:outline-none",placeholder:`# Test 1: Description
input: your_function(args)
expected: expected_result`,spellCheck:!1}),y>0&&p.jsxs("div",{className:"px-4 py-2 bg-gray-800 text-gray-400 text-xs border-t border-gray-700",children:["+ ",y," hidden test",y>1?"s":""," (not editable)"]})]}):p.jsxs("div",{className:"flex-1 overflow-auto p-4 font-mono text-sm bg-gray-900 rounded-b-lg",children:[e.length===0?p.jsx("div",{className:"text-gray-500 italic",children:"Output will appear here when you run your code..."}):e.map((h,g)=>p.jsx("div",{className:`py-0.5 ${h.startsWith("[Error]")?"text-red-400":h.includes("PASSED")?"text-green-400":h.includes("FAILED")?"text-red-400":"text-gray-300"}`,children:h.startsWith("Test")?p.jsxs("span",{children:[h.includes("PASSED")?"✓ ":h.includes("FAILED")?"✗ ":"",h]}):p.jsxs("span",{children:["> ",h]})},g)),p.jsx("div",{ref:o})]})]})}function sz({problem:e}){return p.jsxs("div",{className:"prose prose-sm max-w-none overflow-hidden",children:[p.jsxs("div",{className:"flex items-center gap-3 mb-4",children:[p.jsx("h1",{className:"text-xl font-bold text-gray-900 m-0",children:e.title}),p.jsx("span",{className:`px-2 py-0.5 rounded text-xs font-medium ${e.difficulty==="easy"?"bg-green-100 text-green-700":e.difficulty==="medium"?"bg-yellow-100 text-yellow-700":"bg-red-100 text-red-700"}`,children:e.difficulty})]}),p.jsx(kg,{remarkPlugins:[Qg],components:{h2:({children:t})=>p.jsx("h2",{className:"text-lg font-semibold text-gray-900 mt-6 mb-3",children:t}),h3:({children:t})=>p.jsx("h3",{className:"text-base font-medium text-gray-700 mt-4 mb-2",children:t}),p:({children:t})=>p.jsx("p",{className:"text-gray-600 mb-3 leading-relaxed",children:t}),code:({className:t,children:n})=>(t==null?void 0:t.includes("language-"))?p.jsx("pre",{className:"bg-gray-100 rounded-lg p-4 overflow-x-auto my-3",children:p.jsx("code",{className:"text-sm text-gray-800",children:n})}):p.jsx("code",{className:"bg-gray-100 px-1.5 py-0.5 rounded text-primary-600 text-sm break-words",children:n}),ul:({children:t})=>p.jsx("ul",{className:"list-disc list-inside text-gray-600 space-y-1 mb-3",children:t}),li:({children:t})=>p.jsx("li",{className:"text-gray-600",children:t}),table:({children:t})=>p.jsx("div",{className:"overflow-x-auto my-4",children:p.jsx("table",{className:"min-w-full border-collapse border border-gray-300 text-sm",children:t})}),thead:({children:t})=>p.jsx("thead",{className:"bg-gray-100",children:t}),tbody:({children:t})=>p.jsx("tbody",{className:"divide-y divide-gray-200",children:t}),tr:({children:t})=>p.jsx("tr",{children:t}),th:({children:t})=>p.jsx("th",{className:"border border-gray-300 px-3 py-2 text-left font-semibold text-gray-700",children:t}),td:({children:t})=>p.jsx("td",{className:"border border-gray-300 px-3 py-2 text-gray-600",children:t})},children:e.description})]})}function lz({examples:e}){return p.jsxs("div",{className:"mt-6",children:[p.jsx("h3",{className:"text-base font-medium text-gray-900 mb-3",children:"Examples"}),p.jsx("div",{className:"space-y-4",children:e.map((t,n)=>p.jsx("div",{className:"bg-white rounded-lg p-4 border border-gray-200 shadow-sm",children:p.jsxs("div",{className:"space-y-2",children:[p.jsxs("div",{className:"overflow-hidden",children:[p.jsx("span",{className:"text-gray-500 text-sm",children:"Input:"}),p.jsx("pre",{className:"mt-1 text-sm text-primary-600 font-mono whitespace-pre-wrap break-all",children:t.input})]}),p.jsxs("div",{className:"overflow-hidden",children:[p.jsx("span",{className:"text-gray-500 text-sm",children:"Output:"}),p.jsx("pre",{className:"mt-1 text-sm text-green-600 font-mono whitespace-pre-wrap break-all",children:t.output})]}),t.explanation&&p.jsxs("div",{className:"pt-2 border-t border-gray-200",children:[p.jsx("span",{className:"text-gray-500 text-sm",children:"Explanation: "}),p.jsx("span",{className:"text-gray-600 text-sm",children:t.explanation})]})]})},n))})]})}function uz({hints:e,solution:t}){const[n,r]=E.useState(0),[i,a]=E.useState(!1),o=()=>{n<e.length&&r(l=>l+1)},s=()=>{a(!0)};return p.jsxs("div",{className:"mt-6 space-y-4",children:[p.jsxs("div",{children:[p.jsxs("div",{className:"flex items-center justify-between mb-3",children:[p.jsx("h3",{className:"text-base font-medium text-gray-900",children:"Hints"}),n<e.length&&p.jsxs("button",{onClick:o,className:"text-sm text-primary-600 hover:text-primary-500 transition-colors",children:["Show Hint (",n,"/",e.length,")"]})]}),n>0?p.jsx("div",{className:"space-y-2",children:e.slice(0,n).map((l,u)=>p.jsx("div",{className:"bg-yellow-50 border border-yellow-200 rounded-lg p-3",children:p.jsxs("div",{className:"flex items-start gap-2",children:[p.jsx("span",{className:"text-yellow-600 flex-shrink-0",children:"💡"}),p.jsx("span",{className:"text-gray-700 text-sm break-words overflow-hidden",children:l})]})},u))}):p.jsx("div",{className:"text-gray-400 text-sm italic",children:'Click "Show Hint" if you need help'})]}),p.jsxs("div",{children:[p.jsxs("div",{className:"flex items-center justify-between mb-3",children:[p.jsx("h3",{className:"text-base font-medium text-gray-900",children:"Solution"}),!i&&n>=e.length&&p.jsx("button",{onClick:s,className:"text-sm text-primary-600 hover:text-primary-500 transition-colors",children:"Reveal Solution"})]}),i?p.jsxs("div",{className:"bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm",children:[p.jsx("div",{className:"px-4 py-2 bg-gray-100 border-b border-gray-200",children:p.jsx("span",{className:"text-sm text-gray-600",children:"Solution Code"})}),p.jsx("pre",{className:"p-4 text-sm text-gray-300 font-mono overflow-x-auto bg-gray-900",children:t})]}):n<e.length?p.jsx("div",{className:"text-gray-400 text-sm italic",children:"Reveal all hints first to unlock the solution"}):p.jsx("div",{className:"text-gray-400 text-sm italic",children:'Click "Reveal Solution" to see the answer'})]})]})}function af({results:e,isRunning:t}){if(t)return p.jsxs("div",{className:"flex items-center gap-3 p-4 bg-white rounded-lg border border-gray-200 shadow-sm",children:[p.jsx("div",{className:"w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin"}),p.jsx("span",{className:"text-gray-600",children:"Running tests..."})]});if(e.length===0)return null;const n=e.filter(i=>i.passed).length,r=e.filter(i=>!i.hidden);return p.jsxs("div",{className:"bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm",children:[p.jsxs("div",{className:"px-4 py-3 border-b border-gray-200 flex items-center justify-between bg-gray-50",children:[p.jsx("span",{className:"font-medium text-gray-900",children:"Test Results"}),p.jsxs("span",{className:`text-sm font-medium ${n===e.length?"text-green-600":"text-yellow-600"}`,children:[n,"/",e.length," passed"]})]}),p.jsxs("div",{className:"divide-y divide-gray-200",children:[r.map(i=>p.jsxs("div",{className:"p-4",children:[p.jsxs("div",{className:"flex items-center gap-2 mb-2",children:[p.jsx("span",{className:`w-5 h-5 rounded-full flex items-center justify-center text-xs ${i.passed?"bg-green-100 text-green-600":"bg-red-100 text-red-600"}`,children:i.passed?"✓":"✗"}),p.jsxs("span",{className:"text-gray-700 text-sm font-medium",children:["Test ",i.id,": ",i.description]})]}),!i.passed&&p.jsxs("div",{className:"ml-7 space-y-1 text-sm",children:[p.jsxs("div",{children:[p.jsx("span",{className:"text-gray-500",children:"Expected: "}),p.jsx("span",{className:"text-green-600 font-mono",children:i.expected})]}),p.jsxs("div",{children:[p.jsx("span",{className:"text-gray-500",children:"Actual: "}),p.jsx("span",{className:"text-red-600 font-mono",children:i.actual})]})]})]},i.id)),e.some(i=>i.hidden)&&p.jsxs("div",{className:"p-4 text-gray-500 text-sm italic",children:["+ ",e.filter(i=>i.hidden).length," hidden test(s)",e.filter(i=>i.hidden&&i.passed).length===e.filter(i=>i.hidden).length?p.jsx("span",{className:"text-green-600 ml-2",children:"(all passed)"}):p.jsxs("span",{className:"text-red-600 ml-2",children:["(",e.filter(i=>i.hidden&&!i.passed).length," failed)"]})]})]})]})}function cz(){const{sectionId:e,problemId:t}=Th(),{getProblemProgress:n,saveProblemCode:r,updateProblemStatus:i}=ra(),{isLoading:a,isReady:o,output:s,runTests:l,clearOutput:u}=Jg(),d=t?FC(t):void 0,c=Mr.find(j=>j.id===e),[f,m]=E.useState(""),[y,_]=E.useState([]),[k,h]=E.useState(!1),[g,x]=E.useState(!1),[C,z]=E.useState([]),[S,T]=E.useState(0);E.useEffect(()=>{if(d&&e){const j=n(e,d.id);m(j.code||d.starterCode),z(d.testCases)}},[d,e,n]),E.useEffect(()=>{if(f&&d&&e&&f!==d.starterCode){const j=setTimeout(()=>{r(e,d.id,f)},1e3);return()=>clearTimeout(j)}},[f,d,e,r]);const N=E.useCallback(j=>{m(j)},[]),M=E.useCallback(()=>{d&&(m(d.starterCode),_([]),z(d.testCases),T(j=>j+1),u())},[d,u]),w=j=>{const $=[...j.matchAll(/def\s+(\w+)\s*\(/g)];return $.length>0?$[$.length-1][1]:"solution"},I=E.useCallback(async()=>{if(!(!d||!o||!e)){h(!0),_([]);try{const j=w(d.starterCode),$=await l(f,C,j);_($),$.every(O=>O.passed)&&i(e,d.id,"completed")}finally{h(!1)}}},[d,o,e,f,C,l,i]);if(!d||!c)return p.jsxs("div",{className:"text-center py-12",children:[p.jsx("h1",{className:"text-2xl font-bold text-gray-900 mb-4",children:"Problem Not Found"}),p.jsx(ot,{to:"/",className:"text-primary-600 hover:text-primary-500",children:"Return to Home"})]});const W=c.problems.indexOf(d.id),Y=c.problems[W+1],ee=d.description.replace(/[#*`]/g,"").split(`
`).filter(j=>j.trim()).slice(0,2).join(" ").substring(0,155);return p.jsxs("div",{className:"h-[calc(100vh-8rem)] flex flex-col -m-6",children:[p.jsx(Hr,{title:`${d.title} - ${c.title}`,description:`${ee}... Practice this ${d.difficulty} ML coding problem with instant feedback.`,canonical:`/problem/${e}/${t}`,keywords:`${d.title}, ${c.title}, machine learning, coding practice, ${d.difficulty}`}),p.jsxs("div",{className:"flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200",children:[p.jsxs("div",{className:"flex items-center gap-3",children:[p.jsxs(ot,{to:`/section/${e}`,className:"text-gray-500 hover:text-gray-900 transition-colors",children:["← ",c.title]}),p.jsx("span",{className:"text-gray-300",children:"/"}),p.jsx("span",{className:"text-gray-900 font-medium",children:d.title})]}),p.jsxs("div",{className:"flex items-center gap-3",children:[!o&&p.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-yellow-500 rounded-full animate-pulse"}),a?"Loading Python...":"Python ready"]}),o&&p.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-green-500 rounded-full"}),"Python ready"]})]})]}),g?p.jsxs("div",{className:"flex flex-col flex-1 overflow-hidden bg-white",children:[p.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200",children:[p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("button",{onClick:I,disabled:!o||k,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:k?"Running...":"Run Tests"}),p.jsx("span",{className:"text-xs text-gray-400",children:"or"}),p.jsxs("div",{className:"flex items-center gap-1 text-gray-500",children:[p.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm",children:"Shift"}),p.jsx("span",{className:"text-xs",children:"+"}),p.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm",children:"Enter"})]})]}),p.jsxs("div",{className:"flex items-center gap-2",children:[y.length>0&&y.every(j=>j.passed)&&Y&&p.jsx(ot,{to:`/problem/${e}/${Y}`,className:"px-4 py-1.5 bg-green-100 text-green-700 text-sm font-medium rounded-md hover:bg-green-200 transition-colors",children:"Next Problem →"}),p.jsx("button",{onClick:()=>x(!g),className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",title:g?"Restore layout":"Maximize editor",children:g?"⊟ Restore":"⊞ Maximize"}),p.jsx("button",{onClick:M,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Reset"})]})]}),p.jsxs(Nr,{className:"flex flex-1 overflow-hidden",sizes:[60,40],minSize:200,gutterSize:8,gutterAlign:"center",children:[p.jsx("div",{className:"h-full overflow-hidden p-4",children:p.jsx(pu,{value:f,onChange:N,height:"100%",onRun:I})}),p.jsxs("div",{className:"overflow-auto p-4 space-y-4 bg-gray-50",children:[p.jsx(af,{results:y,isRunning:k}),p.jsx(fu,{output:s,isLoading:k,testCases:C,onTestCasesChange:z,problemId:t,resetKey:S})]})]})]}):p.jsxs(Nr,{className:"flex flex-1 overflow-hidden",sizes:[40,60],minSize:300,gutterSize:8,gutterAlign:"center",children:[p.jsxs("div",{className:"problem-panel overflow-y-auto overflow-x-hidden p-6 bg-gray-50 min-w-0",children:[p.jsx(sz,{problem:d}),p.jsx(lz,{examples:d.examples}),p.jsx(uz,{hints:d.hints,solution:d.solution})]}),p.jsxs("div",{className:"flex flex-col overflow-hidden bg-white",children:[p.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200",children:[p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("button",{onClick:I,disabled:!o||k,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:k?"Running...":"Run Tests"}),p.jsx("span",{className:"text-xs text-gray-400",children:"or"}),p.jsxs("div",{className:"flex items-center gap-1 text-gray-500",children:[p.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm",children:"Shift"}),p.jsx("span",{className:"text-xs",children:"+"}),p.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm",children:"Enter"})]})]}),p.jsxs("div",{className:"flex items-center gap-2",children:[y.length>0&&y.every(j=>j.passed)&&Y&&p.jsx(ot,{to:`/problem/${e}/${Y}`,className:"px-4 py-1.5 bg-green-100 text-green-700 text-sm font-medium rounded-md hover:bg-green-200 transition-colors",children:"Next Problem →"}),p.jsx("button",{onClick:()=>x(!g),className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",title:g?"Restore layout":"Maximize editor",children:g?"⊟ Restore":"⊞ Maximize"}),p.jsx("button",{onClick:M,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Reset"})]})]}),p.jsxs(Nr,{className:"flex-1 overflow-hidden",direction:"vertical",sizes:[60,40],minSize:100,gutterSize:8,children:[p.jsx("div",{className:"h-full overflow-hidden p-4",children:p.jsx(pu,{value:f,onChange:N,height:"100%",onRun:I})}),p.jsxs("div",{className:"overflow-auto p-4 space-y-4 bg-gray-50",children:[p.jsx(af,{results:y,isRunning:k}),p.jsx(fu,{output:s,isLoading:k,testCases:C,onTestCasesChange:z,problemId:t,resetKey:S})]})]})]})]})]})}const of="ml-interview-scratchpad",sf=`import numpy as np

# Try out your code here!
# NumPy is already loaded and ready to use.

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {np.sum(arr)}")
print(f"Mean: {np.mean(arr)}")
`;function dz(){const{isLoading:e,isReady:t,output:n,runCode:r,clearOutput:i}=Jg(),[a,o]=E.useState(""),[s,l]=E.useState(!1);E.useEffect(()=>{const f=localStorage.getItem(of);o(f||sf)},[]),E.useEffect(()=>{if(a){const f=setTimeout(()=>{localStorage.setItem(of,a)},1e3);return()=>clearTimeout(f)}},[a]);const u=E.useCallback(f=>{o(f)},[]),d=E.useCallback(()=>{o(sf),i()},[i]),c=E.useCallback(async()=>{if(t){l(!0),i();try{await r(a)}finally{l(!1)}}},[t,a,r,i]);return p.jsxs("div",{className:"h-[calc(100vh-8rem)] flex flex-col -m-6",children:[p.jsx(Hr,{title:"Python Scratchpad",description:"Free Python playground with NumPy pre-loaded. Run Python code directly in your browser - no setup required. Practice machine learning code instantly.",canonical:"/scratchpad",keywords:"Python playground, online Python, NumPy, code sandbox, ML practice, browser Python"}),p.jsxs("div",{className:"flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200",children:[p.jsxs("div",{className:"flex items-center gap-3",children:[p.jsx(ot,{to:"/",className:"text-gray-500 hover:text-gray-900 transition-colors",children:"Home"}),p.jsx("span",{className:"text-gray-300",children:"/"}),p.jsx("span",{className:"text-gray-900 font-medium",children:"Python Scratchpad"})]}),p.jsxs("div",{className:"flex items-center gap-3",children:[!t&&p.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-yellow-500 rounded-full animate-pulse"}),e?"Loading Python...":"Python ready"]}),t&&p.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-green-500 rounded-full"}),"Python ready"]})]})]}),p.jsxs("div",{className:"flex flex-col flex-1 overflow-hidden bg-white",children:[p.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200",children:[p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("button",{onClick:c,disabled:!t||s,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:s?"Running...":"Run Code"}),p.jsx("button",{onClick:d,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Reset"}),p.jsx("button",{onClick:i,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Clear Output"})]}),p.jsx("span",{className:"text-gray-400 text-sm",children:"Code is auto-saved"})]}),p.jsxs(Nr,{className:"flex-1 overflow-hidden",direction:"vertical",sizes:[60,40],minSize:100,gutterSize:8,children:[p.jsx("div",{className:"h-full overflow-hidden p-4",children:p.jsx(pu,{value:a,onChange:u,height:"100%"})}),p.jsx("div",{className:"overflow-auto p-4 bg-gray-50",children:p.jsx(fu,{output:n,isLoading:s})})]})]})]})}function pz(){return p.jsxs(p.Fragment,{children:[p.jsx(Hr,{title:"Terms of Service",description:"Terms of Service for ML Coding Lab - an interactive platform for learning machine learning through hands-on coding exercises.",canonical:"/terms"}),p.jsxs("div",{className:"max-w-4xl mx-auto",children:[p.jsx("h1",{className:"text-3xl font-bold text-gray-900 mb-8",children:"Terms of Service"}),p.jsxs("div",{className:"prose prose-gray max-w-none",children:[p.jsxs("p",{className:"text-gray-600 mb-6",children:[p.jsx("strong",{children:"Last updated:"})," February 1, 2026"]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"1. Acceptance of Terms"}),p.jsx("p",{className:"text-gray-600 mb-4",children:'By accessing and using ML Coding Lab ("the Service"), you accept and agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use the Service.'})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"2. Description of Service"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"ML Coding Lab is an educational platform that provides interactive machine learning coding exercises, tutorials, and learning materials. The Service allows users to:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Read problem descriptions and theoretical introductions"}),p.jsx("li",{children:"Write and execute Python code in the browser"}),p.jsx("li",{children:"Run tests and receive feedback on solutions"}),p.jsx("li",{children:"Track learning progress across sections"}),p.jsx("li",{children:"Use a Python scratchpad for experimentation"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"3. Intellectual Property Rights"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"Software Code:"})," The source code of ML Coding Lab is licensed under the MIT License. You may use, modify, and distribute the code in accordance with the terms of that license."]}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"Educational Content:"})," All educational content, including but not limited to:"]}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Problem descriptions and explanations"}),p.jsx("li",{children:"Solutions and starter code for exercises"}),p.jsx("li",{children:"Learning materials and tutorials"}),p.jsx("li",{children:"Test cases and examples"})]}),p.jsx("p",{className:"text-gray-600 mb-4",children:"is copyrighted material owned by Siddharth Choudhary. This content may not be reproduced, distributed, modified, or used to create derivative works without explicit written permission from the copyright holder."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"4. Permitted Use"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"You may use the Service for:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Personal learning and educational purposes"}),p.jsx("li",{children:"Preparing for interviews or assessments"}),p.jsx("li",{children:"Building your machine learning skills"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"5. Prohibited Use"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"You may NOT:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Copy, reproduce, or redistribute the educational content"}),p.jsx("li",{children:"Use the content for commercial purposes without permission"}),p.jsx("li",{children:"Create competing services using our educational materials"}),p.jsx("li",{children:"Scrape or bulk download content from the Service"}),p.jsx("li",{children:"Remove or alter any copyright notices or attributions"}),p.jsx("li",{children:"Attempt to circumvent any security measures"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"6. User Code"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"Code that you write in the code editor is stored locally in your browser. We do not collect or store your code on our servers. You retain full ownership of any original code you create while using the Service."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"7. Disclaimer of Warranties"}),p.jsx("p",{className:"text-gray-600 mb-4",children:'THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED. WE DO NOT WARRANT THAT THE SERVICE WILL BE UNINTERRUPTED, ERROR-FREE, OR SECURE.'})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"8. Limitation of Liability"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"TO THE MAXIMUM EXTENT PERMITTED BY LAW, WE SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING FROM YOUR USE OF THE SERVICE."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"9. Changes to Terms"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"We reserve the right to modify these Terms at any time. Changes will be effective immediately upon posting. Your continued use of the Service after changes constitutes acceptance of the modified Terms."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"10. Contact"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:["For questions about these Terms, please open an issue on our"," ",p.jsx("a",{href:"https://github.com/itzsid/ml-coding-lab",target:"_blank",rel:"noopener noreferrer",className:"text-primary-600 hover:text-primary-700",children:"GitHub repository"}),"."]})]})]})]})]})}function fz(){return p.jsxs(p.Fragment,{children:[p.jsx(Hr,{title:"Privacy Policy",description:"Privacy Policy for ML Coding Lab - learn how we handle your data on our machine learning education platform.",canonical:"/privacy"}),p.jsxs("div",{className:"max-w-4xl mx-auto",children:[p.jsx("h1",{className:"text-3xl font-bold text-gray-900 mb-8",children:"Privacy Policy"}),p.jsxs("div",{className:"prose prose-gray max-w-none",children:[p.jsxs("p",{className:"text-gray-600 mb-6",children:[p.jsx("strong",{children:"Last updated:"})," February 1, 2026"]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Overview"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"ML Coding Lab is committed to protecting your privacy. This Privacy Policy explains how we collect, use, and protect information when you use our Service."}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"The short version:"})," We collect minimal data. Your code stays in your browser. We don't track you or sell your data."]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Information We Collect"}),p.jsx("h3",{className:"text-lg font-medium text-gray-700 mb-3",children:"Data Stored Locally (Your Browser)"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"The following data is stored only in your browser's localStorage and never sent to our servers:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsxs("li",{children:[p.jsx("strong",{children:"Progress data:"})," Which problems you've completed"]}),p.jsxs("li",{children:[p.jsx("strong",{children:"Saved code:"})," Your code solutions for each problem"]}),p.jsxs("li",{children:[p.jsx("strong",{children:"Scratchpad code:"})," Code you've written in the Python scratchpad"]})]}),p.jsx("p",{className:"text-gray-600 mb-4",children:"This data never leaves your device unless you explicitly export it."}),p.jsx("h3",{className:"text-lg font-medium text-gray-700 mb-3",children:"Feedback Submissions"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"When you submit feedback through our feedback form, we collect:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Your email address (optional, only if you provide it)"}),p.jsx("li",{children:"The feedback message you write"})]}),p.jsx("p",{className:"text-gray-600 mb-4",children:"This information is sent via Formspree and is used solely to respond to your feedback and improve the Service."}),p.jsx("h3",{className:"text-lg font-medium text-gray-700 mb-3",children:"Analytics and Hosting"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:["ML Coding Lab is hosted on GitHub Pages. GitHub may collect standard server logs including IP addresses and page requests. Please refer to"," ",p.jsx("a",{href:"https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement",target:"_blank",rel:"noopener noreferrer",className:"text-primary-600 hover:text-primary-700",children:"GitHub's Privacy Statement"})," ","for more information."]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Information We Do NOT Collect"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"We do not require account registration"}),p.jsx("li",{children:"We do not use cookies for tracking"}),p.jsx("li",{children:"We do not use third-party analytics services"}),p.jsx("li",{children:"We do not collect personal information beyond what you voluntarily provide"}),p.jsx("li",{children:"We do not sell or share your data with third parties"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Code Execution"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"All Python code execution happens entirely in your browser using Pyodide (WebAssembly). Your code is never sent to any external server for execution. This means:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Your code remains private on your device"}),p.jsx("li",{children:"We cannot see or access the code you write"}),p.jsx("li",{children:"Execution results stay in your browser"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Data Retention"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"Local data:"})," Data stored in your browser persists until you clear your browser's localStorage or use a browser cleaning tool."]}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"Feedback:"})," Feedback submissions are retained only as long as necessary to respond to your inquiry and improve the Service."]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Your Rights"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"You have the right to:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Clear your local data at any time through your browser settings"}),p.jsx("li",{children:"Use the Service without providing any personal information"}),p.jsx("li",{children:"Request deletion of any feedback you've submitted"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Children's Privacy"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"The Service is intended for general audiences interested in learning machine learning. We do not knowingly collect personal information from children under 13. Since we don't require registration and store data locally, children can use the Service without providing any personal information."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Changes to This Policy"}),p.jsx("p",{className:"text-gray-600 mb-4",children:'We may update this Privacy Policy from time to time. We will notify users of any material changes by updating the "Last updated" date at the top of this policy.'})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Contact"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:["For questions about this Privacy Policy, please open an issue on our"," ",p.jsx("a",{href:"https://github.com/itzsid/ml-coding-lab",target:"_blank",rel:"noopener noreferrer",className:"text-primary-600 hover:text-primary-700",children:"GitHub repository"}),"."]})]})]})]})]})}function mz(){return p.jsx(P_,{children:p.jsxs(R1,{children:[p.jsx(Mn,{path:"/",element:p.jsx(A_,{})}),p.jsx(Mn,{path:"/section/:sectionId",element:p.jsx(qC,{})}),p.jsx(Mn,{path:"/problem/:sectionId/:problemId",element:p.jsx(cz,{})}),p.jsx(Mn,{path:"/scratchpad",element:p.jsx(dz,{})}),p.jsx(Mn,{path:"/terms",element:p.jsx(pz,{})}),p.jsx(Mn,{path:"/privacy",element:p.jsx(fz,{})})]})})}Js.createRoot(document.getElementById("root")).render(p.jsx(Ne.StrictMode,{children:p.jsx(Mh,{children:p.jsx(B1,{basename:"/ml-coding-lab",children:p.jsx(S_,{children:p.jsx(mz,{})})})})}));
